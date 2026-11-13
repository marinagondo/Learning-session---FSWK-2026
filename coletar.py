# -*- coding: utf-8 -*-
"""
coletar.py - Robô de coleta de dados do DATASUS.

O que faz:
  1. Conecta no FTP do DATASUS e procura os arquivos .DBC de cada estado/ano.
  2. Descompacta, lê só as colunas necessárias e conta os registros por município.
  3. Salva arquivos .parquet pequenos em saida/<base>/.
  4. Com --consolidar, junta tudo em dados/<base>/ (é isso que o dashboard lê).

Uso:
  python coletar.py --base sinasc --uf RJ        (coleta um estado)
  python coletar.py --base sinasc --uf RJ,SP     (vários estados)
  python coletar.py --base sinasc --uf TODAS     (todos os estados)
  python coletar.py --consolidar                 (junta os resultados)
"""
import argparse
import ftplib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------------
ANO_INICIAL = 2010
ANO_FINAL = 2024
FTP_HOST = "ftp.datasus.gov.br"
PASTA_DADOS = "dados"    # resultado final (vai para o repositório)
PASTA_SAIDA = "saida"    # resultado intermediário, por estado e ano

# código IBGE da UF -> sigla
UF_SIGLAS = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}
SIGLA_PARA_COD = {sigla: cod for cod, sigla in UF_SIGLAS.items()}

FTP_TIMEOUT = 180
FTP_TENTATIVAS = 5
BLOCO_REGISTROS = 400_000


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# FTP (modo passivo, reconexão, tentativas com espera crescente, cache)
# ----------------------------------------------------------------------------
class ClienteFTP:
    def __init__(self, host=FTP_HOST):
        self.host = host
        self.ftp = None
        self.cache_listagem = {}  # pasta -> {NOME_MAIUSCULO: (nome_real, tamanho)}

    def _conectar(self):
        self.fechar()
        ftp = ftplib.FTP(timeout=FTP_TIMEOUT)
        ftp.connect(self.host, 21)
        ftp.login()  # anônimo
        ftp.set_pasv(True)
        self.ftp = ftp

    def fechar(self):
        if self.ftp is not None:
            try:
                self.ftp.quit()
            except Exception:
                try:
                    self.ftp.close()
                except Exception:
                    pass
            self.ftp = None

    def _com_tentativas(self, funcao, descricao):
        ultimo_erro = None
        for tentativa in range(FTP_TENTATIVAS):
            try:
                if self.ftp is None:
                    self._conectar()
                return funcao(self.ftp)
            except (ftplib.all_errors, OSError, EOFError) as erro:
                ultimo_erro = erro
                espera = 5 * (2 ** tentativa)
                log(f"  ! Falha em {descricao} ({erro}). "
                    f"Nova tentativa {tentativa + 1}/{FTP_TENTATIVAS} em {espera}s")
                self.fechar()
                time.sleep(espera)
        raise RuntimeError(f"Desisti de {descricao}: {ultimo_erro}")

    @staticmethod
    def _interpretar_linha(linha):
        """Aceita listagem estilo Windows (DATASUS) e estilo Unix."""
        partes = linha.split()
        if len(partes) >= 4 and partes[2].isdigit():           # 04-12-23 10:15AM 123 NOME
            return " ".join(partes[3:]), int(partes[2])
        if len(partes) >= 9 and partes[4].isdigit():           # -rw-r--r-- 1 u g 123 mes d h NOME
            return " ".join(partes[8:]), int(partes[4])
        return None

    def listar(self, pasta):
        """Retorna {NOME_MAIUSCULO: (nome_real, tamanho)} (com cache)."""
        if pasta in self.cache_listagem:
            return self.cache_listagem[pasta]

        def _listar(ftp):
            linhas = []
            ftp.retrlines(f"LIST {pasta}", linhas.append)
            return linhas

        try:
            linhas = self._com_tentativas(_listar, f"listar {pasta}")
        except RuntimeError as erro:
            log(f"  ! Pasta indisponível: {pasta} ({erro})")
            self.cache_listagem[pasta] = {}
            return {}
        itens = {}
        for linha in linhas:
            achado = self._interpretar_linha(linha)
            if achado:
                nome, tamanho = achado
                itens[nome.upper()] = (nome, tamanho)
        self.cache_listagem[pasta] = itens
        return itens

    def baixar(self, pasta, nome_real, destino):
        def _baixar(ftp):
            with open(destino, "wb") as saida:
                ftp.retrbinary(f"RETR {pasta}/{nome_real}", saida.write)

        self._com_tentativas(_baixar, f"baixar {nome_real}")


# ----------------------------------------------------------------------------
# DBC -> DBF
# ----------------------------------------------------------------------------
def descompactar_dbc(caminho_dbc, caminho_dbf):
    from pyreaddbc import dbc2dbf  # importado aqui para o resto funcionar sem ele
    try:
        dbc2dbf(str(caminho_dbc), str(caminho_dbf))
    except TypeError:
        dbc2dbf(str(caminho_dbc).encode(), str(caminho_dbf).encode())
    if not os.path.exists(caminho_dbf) or os.path.getsize(caminho_dbf) < 32:
        raise RuntimeError("Falha ao descompactar o arquivo DBC")


# ----------------------------------------------------------------------------
# Leitor DBF próprio (rápido, só as colunas pedidas, em blocos)
# ----------------------------------------------------------------------------
def ler_cabecalho_dbf(caminho):
    with open(caminho, "rb") as f:
        cab = f.read(32)
        n_registros = int.from_bytes(cab[4:8], "little")
        tam_cabecalho = int.from_bytes(cab[8:10], "little")
        tam_registro = int.from_bytes(cab[10:12], "little")
        campos = {}  # NOME -> (deslocamento dentro do registro, tamanho)
        deslocamento = 1  # byte 0 do registro é o marcador de apagado
        while True:
            desc = f.read(32)
            if len(desc) < 32 or desc[0] == 0x0D:
                break
            nome = desc[:11].split(b"\x00")[0].decode("latin-1").strip().upper()
            tamanho = desc[16]
            campos[nome] = (deslocamento, tamanho)
            deslocamento += tamanho
    return n_registros, tam_cabecalho, tam_registro, campos


def ler_dbf_em_blocos(caminho, colunas, tamanho_bloco=BLOCO_REGISTROS):
    """
    Gera dicionários {coluna: np.array de str} com até `tamanho_bloco` registros.
    Registros apagados ('*') são ignorados. Colunas inexistentes vêm vazias.
    """
    n_reg, tam_cab, tam_reg, campos = ler_cabecalho_dbf(caminho)
    if n_reg == 0:
        return
    dados = np.memmap(caminho, dtype=np.uint8, mode="r", offset=tam_cab,
                      shape=(n_reg, tam_reg))
    try:
        for ini in range(0, n_reg, tamanho_bloco):
            bloco = dados[ini:ini + tamanho_bloco]
            vivos = bloco[:, 0] != ord("*")
            resultado = {}
            for col in colunas:
                if col not in campos:
                    resultado[col] = np.full(int(vivos.sum()), "", dtype=object)
                    continue
                desloc, tam = campos[col]
                fatia = np.ascontiguousarray(bloco[:, desloc:desloc + tam])
                bruto = fatia.view(f"S{tam}").reshape(-1)[vivos]
                texto = np.char.strip(np.char.decode(bruto, "latin-1"))
                resultado[col] = texto.astype(object)
            yield resultado
    finally:
        del dados


# ----------------------------------------------------------------------------
# SINASC - regras de agregação
# ----------------------------------------------------------------------------
SINASC_COLUNAS = ["CODMUNRES", "SEXO", "RACACOR", "IDADEMAE", "ESCMAE"]

MAPA_SEXO = {"1": "Masculino", "M": "Masculino", "2": "Feminino", "F": "Feminino"}
MAPA_RACA = {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}
MAPA_ESCOLARIDADE = {
    "1": "Nenhuma", "2": "1 a 3 anos", "3": "4 a 7 anos",
    "4": "8 a 11 anos", "5": "12 anos ou mais",
}
FAIXAS_IDADE = ["Menos de 15", "15 a 19", "20 a 24", "25 a 29",
                "30 a 34", "35 a 39", "40 ou mais"]


def limpar_codigo_municipio(serie, cod_uf):
    """6 dígitos; inválido (ou de outra UF) vira '<UF>0000' (município ignorado)."""
    s = serie.astype(str).str.strip()
    valido = s.str.fullmatch(r"\d{6,7}") & (s.str[:2] == cod_uf)
    return s.str[:6].where(valido, cod_uf + "0000")


def _contar(cod_mun, categoria, nome_variavel):
    df = pd.DataFrame({"cod_mun": cod_mun.to_numpy(),
                       "categoria": pd.Series(categoria).to_numpy()})
    df = df.dropna(subset=["categoria"])
    if df.empty:
        return df.assign(variavel=nome_variavel, n=0)[["cod_mun", "variavel", "categoria", "n"]]
    g = df.groupby(["cod_mun", "categoria"], sort=False).size().reset_index(name="n")
    g["variavel"] = nome_variavel
    return g[["cod_mun", "variavel", "categoria", "n"]]


def agregar_sinasc(cols, cod_uf):
    cod = limpar_codigo_municipio(pd.Series(cols["CODMUNRES"]), cod_uf)
    partes = []

    partes.append(_contar(cod, pd.Series("Total", index=cod.index), "total"))

    sexo = pd.Series(cols["SEXO"]).str.upper().map(MAPA_SEXO).fillna("Ignorado")
    partes.append(_contar(cod, sexo, "sexo"))

    raca = pd.Series(cols["RACACOR"]).map(MAPA_RACA).fillna("Ignorado")
    partes.append(_contar(cod, raca, "raca_cor"))

    idade = pd.to_numeric(pd.Series(cols["IDADEMAE"]), errors="coerce")
    faixa = pd.cut(idade, bins=[10, 15, 20, 25, 30, 35, 40, 61], right=False,
                   labels=FAIXAS_IDADE).astype(object)  # idade inválida -> NaN -> ignorada
    partes.append(_contar(cod, faixa, "faixa_idade_mae"))

    esc = pd.Series(cols["ESCMAE"]).map(MAPA_ESCOLARIDADE).fillna("Ignorado")
    partes.append(_contar(cod, esc, "escolaridade_mae"))

    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------------
# Configuração das bases (novas bases entram aqui nas próximas etapas)
# ----------------------------------------------------------------------------
BASES = {
    "sinasc": {
        "nome_arquivo": lambda uf, ano: f"DN{uf}{ano}.DBC",
        "pastas": [  # (pasta, é_preliminar) - na ordem de preferência
            ("/dissemin/publicos/SINASC/NOV/DNRES", False),
            ("/dissemin/publicos/SINASC/1996_/Dados/DNRES", False),
            ("/dissemin/publicos/SINASC/PRELIM/DNRES", True),
        ],
        "colunas": SINASC_COLUNAS,
        "agregar": agregar_sinasc,
    },
}


# ----------------------------------------------------------------------------
# Utilitários de arquivos
# ----------------------------------------------------------------------------
def caminho_parquet(base, uf, ano, prelim, por_uf=False):
    sufixo = "_prelim" if prelim else ""
    pasta = os.path.join(PASTA_SAIDA, base, "_uf") if por_uf else os.path.join(PASTA_SAIDA, base)
    return os.path.join(pasta, f"{uf}_{ano}{sufixo}.parquet")


def caminho_controle(base, uf):
    return os.path.join(PASTA_DADOS, base, "_controle", f"{uf}.json")


def ler_controle(base, uf):
    try:
        with open(caminho_controle(base, uf), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def salvar_controle(base, uf, controle):
    caminho = caminho_controle(base, uf)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(controle, f, indent=1, sort_keys=True)


def salvar_parquet(df, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df = df.copy()
    df["ano"] = df["ano"].astype("int16")
    df["n"] = df["n"].astype("int32")
    df[["cod_mun", "variavel", "categoria"]] = df[["cod_mun", "variavel", "categoria"]].astype(str)
    df.to_parquet(caminho, index=False, compression="zstd")


def localizar_arquivo(ftp, cfg, uf, ano):
    """Procura o arquivo nas pastas, na ordem. Retorna (pasta, nome_real, tamanho, prelim) ou None."""
    nome = cfg["nome_arquivo"](uf, ano).upper()
    for pasta, prelim in cfg["pastas"]:
        itens = ftp.listar(pasta)
        if nome in itens:
            nome_real, tamanho = itens[nome]
            return pasta, nome_real, tamanho, prelim
    return None


# ----------------------------------------------------------------------------
# Coleta de um estado
# ----------------------------------------------------------------------------
def processar_arquivo(ftp, cfg, uf, ano, pasta, nome_real, prelim, base):
    cod_uf = SIGLA_PARA_COD[uf]
    tmp = tempfile.mkdtemp(prefix=f"{base}_{uf}_{ano}_")
    try:
        dbc = os.path.join(tmp, "arquivo.dbc")
        dbf = os.path.join(tmp, "arquivo.dbf")
        ftp.baixar(pasta, nome_real, dbc)
        descompactar_dbc(dbc, dbf)
        os.remove(dbc)
        partes = []
        for bloco in ler_dbf_em_blocos(dbf, cfg["colunas"]):
            partes.append(cfg["agregar"](bloco, cod_uf))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not partes:
        return None
    df = pd.concat(partes, ignore_index=True)
    df = df.groupby(["cod_mun", "variavel", "categoria"], sort=True, as_index=False)["n"].sum()
    df.insert(0, "ano", ano)
    return df[["ano", "cod_mun", "variavel", "categoria", "n"]]


def coletar_uf(base, uf, ftp):
    cfg = BASES[base]
    cod_uf = SIGLA_PARA_COD[uf]
    controle = ler_controle(base, uf)
    falhas = 0
    log(f"=== {base.upper()} | {uf} ===")

    for ano in range(ANO_INICIAL, ANO_FINAL + 1):
        achado = localizar_arquivo(ftp, cfg, uf, ano)
        if achado is None:
            log(f"  {uf} {ano}: não encontrado no FTP")
            continue
        pasta, nome_real, tamanho, prelim = achado
        assinatura = f"{pasta}/{nome_real}:{tamanho}"
        destino = caminho_parquet(base, uf, ano, prelim)
        if controle.get(str(ano)) == assinatura and os.path.exists(destino):
            log(f"  {uf} {ano}: sem mudanças, pulando")
            continue

        inicio = time.time()
        try:
            df = processar_arquivo(ftp, cfg, uf, ano, pasta, nome_real, prelim, base)
        except Exception as erro:  # um ano com problema não derruba os outros
            falhas += 1
            log(f"  {uf} {ano}: ERRO ({erro})")
            continue
        if df is None or df.empty:
            log(f"  {uf} {ano}: arquivo sem registros")
            continue

        salvar_parquet(df, destino)
        df_uf = df.groupby(["ano", "variavel", "categoria"], as_index=False)["n"].sum()
        df_uf.insert(1, "cod_mun", cod_uf)
        df_uf = df_uf[["ano", "cod_mun", "variavel", "categoria", "n"]]
        salvar_parquet(df_uf, caminho_parquet(base, uf, ano, prelim, por_uf=True))

        # se a versão final apareceu, remove a preliminar antiga
        if not prelim:
            for por_uf in (False, True):
                antigo = caminho_parquet(base, uf, ano, True, por_uf=por_uf)
                if os.path.exists(antigo):
                    os.remove(antigo)

        controle[str(ano)] = assinatura
        salvar_controle(base, uf, controle)
        total = int(df[df["variavel"] == "total"]["n"].sum())
        etiqueta = "preliminar" if prelim else "final"
        log(f"  {uf} {ano}: {total:,} registros ({etiqueta}) em {time.time() - inicio:.0f}s".replace(",", "."))

    return falhas


# ----------------------------------------------------------------------------
# Consolidação
# ----------------------------------------------------------------------------
def _ler_pasta(pasta):
    if not os.path.isdir(pasta):
        return [], []
    arquivos = sorted(f for f in os.listdir(pasta) if f.endswith(".parquet"))
    dfs = [pd.read_parquet(os.path.join(pasta, f)) for f in arquivos]
    return arquivos, dfs


def consolidar():
    if not os.path.isdir(PASTA_SAIDA):
        log("Nada para consolidar (pasta 'saida' não existe).")
        return 1
    algum = False
    for base in sorted(os.listdir(PASTA_SAIDA)):
        pasta_base = os.path.join(PASTA_SAIDA, base)
        if not os.path.isdir(pasta_base):
            continue
        arquivos_mun, dfs_mun = _ler_pasta(pasta_base)
        arquivos_uf, dfs_uf = _ler_pasta(os.path.join(pasta_base, "_uf"))
        if not dfs_mun:
            continue
        algum = True
        destino = os.path.join(PASTA_DADOS, base)
        os.makedirs(destino, exist_ok=True)

        chaves = ["ano", "cod_mun", "variavel", "categoria"]
        mun = pd.concat(dfs_mun, ignore_index=True).sort_values(chaves)
        mun.to_parquet(os.path.join(destino, "municipios.parquet"), index=False, compression="zstd")
        if dfs_uf:
            uf = pd.concat(dfs_uf, ignore_index=True).sort_values(chaves)
            uf.to_parquet(os.path.join(destino, "uf.parquet"), index=False, compression="zstd")

        prelim = sorted({int(re.search(r"_(\d{4})_prelim", f).group(1))
                         for f in arquivos_mun if "_prelim" in f})
        meta = {
            "base": base,
            "atualizado_em": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "ano_min": int(mun["ano"].min()),
            "ano_max": int(mun["ano"].max()),
            "anos_preliminares": prelim,
            "estados": sorted({f.split("_")[0] for f in arquivos_mun}),
            "linhas": int(len(mun)),
        }
        with open(os.path.join(destino, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log(f"Consolidado {base}: {len(mun):,} linhas, {len(meta['estados'])} estados, "
            f"{meta['ano_min']}-{meta['ano_max']}".replace(",", "."))
    if not algum:
        log("Nenhum arquivo .parquet encontrado em 'saida'.")
        return 1
    return 0


# ----------------------------------------------------------------------------
# Linha de comando
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Robô de coleta do DATASUS")
    ap.add_argument("--base", default="sinasc", choices=sorted(BASES))
    ap.add_argument("--uf", help="Sigla do estado (ex.: RJ), lista (RJ,SP) ou TODAS")
    ap.add_argument("--consolidar", action="store_true", help="Junta os resultados em dados/")
    args = ap.parse_args(argv)

    if args.consolidar:
        return consolidar()
    if not args.uf:
        ap.error("informe --uf (ex.: --uf RJ) ou --consolidar")

    if args.uf.strip().upper() == "TODAS":
        ufs = sorted(SIGLA_PARA_COD)
    else:
        ufs = [u.strip().upper() for u in args.uf.split(",") if u.strip()]
    invalidas = [u for u in ufs if u not in SIGLA_PARA_COD]
    if invalidas:
        ap.error(f"UF inválida: {', '.join(invalidas)}")

    ftp = ClienteFTP()
    falhas = 0
    try:
        for uf in ufs:
            falhas += coletar_uf(args.base, uf, ftp)
    finally:
        ftp.fechar()
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
