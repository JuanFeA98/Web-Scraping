import argparse
import hashlib
import nltk
from nltk.corpus import stopwords
import logging
logging.basicConfig(level=logging.INFO)
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

def main(filename):
    logger.info('Empezando el proceso de limpieza')

    df = _read_data(filename)
    newspaper_uid = _extract_newspaper_uid(filename)
    df = _add_newspaper_uid_column(df, newspaper_uid)
    df = _extract_host(df)
    df = _fill_missing_titles(df)
    df = _generate_uids_for_rows(df)
    df = _remove_new_lines_title(df)
    df = _remove_new_lines_body(df)
    df = _generate_stop_words(df)
    df = _remove_duplicated_entries(df, 'title')
    df = _drop_rows_with_missing_values(df)
    _save_data(df, filename)

    return df


def _read_data(filename):
    logger.info(f'Leyendo el archivo: {filename}')

    return pd.read_csv(filename, encoding = 'utf-8')


def _extract_newspaper_uid(filename):
    logger.info(f'Leyendo archivo {filename}')
    newspaper_uid = filename.split('_')[0]

    logger.info(f'Newspaper uid ha sido detectado: {newspaper_uid}')
    return newspaper_uid


def _add_newspaper_uid_column(df, newspaper_uid):
    logger.info(f'Llenando la columna newspaper_uid con {newspaper_uid}')
    df['newspaper_uid'] = newspaper_uid

    return df


def _extract_host(df):
    logger.info(f'Extrayendo el host de los enlaces')
    df['host'] = df['url'].apply(lambda url: urlparse(url).netloc)

    return df


def _fill_missing_titles(df):
    logger.info(f'Llenando titulos vacios')

    missing_titles_mask = df['title'].isna()

    missing_titles = (df[missing_titles_mask]['url']
                        .str.extract(r'(?P<missing_titles>[^/]+)$')
                        .applymap(lambda title : title.replace('-', ' '))
                        .applymap(lambda title : title.capitalize())
                        )

    df.loc[missing_titles_mask, 'title'] = missing_titles.loc[:, 'missing_titles']

    return df


def _remove_new_lines_title(df):
    logging.info('Eliminando saltos de linea en el titulo')

    stripped_title = (df
                    .apply(lambda row: row['title'], axis=1)
                    .apply(lambda title: list(title))
                    .apply(lambda letters: list(map(lambda letter: letter.replace('\n', ''), letters)))
                    .apply(lambda letters: list(map(lambda letter: letter.replace('\r', ''), letters)))
                    .apply(lambda letters_list: ''.join(letters_list))
                    )

    df['title'] = stripped_title

    return df 


def _remove_new_lines_body(df):
    logging.info('Eliminando saltos de linea en el cuerpo')

    stripped_body = (df
                    .apply(lambda row: row['body'], axis=1)
                    .apply(lambda body: list(body))
                    .apply(lambda letters: list(map(lambda letter: letter.replace('\n', ''), letters)))
                    .apply(lambda letters: list(map(lambda letter: letter.replace('\r', ''), letters)))
                    .apply(lambda letters_list: ''.join(letters_list))
                    )

    df['body'] = stripped_body

    return df 


def _generate_uids_for_rows(df):
    logger.info(f'Creando uids para cada fila')

    uids = (df
            .apply(lambda row: hashlib.md5(bytes(row['url'].encode())), axis = 1)
            .apply(lambda hash_object: hash_object.hexdigest())
            )

    df['uid'] = uids

    return df.set_index('uid')


def tokenize_column(df, column_name):

    stop_words = set(stopwords.words('spanish'))

    return(df
            .dropna()
            .apply(lambda row : nltk.word_tokenize(row[column_name]), axis=1)
            .apply(lambda tokens : list(filter(lambda token : token.isalpha(), tokens)))
            .apply(lambda tokens : list(map(lambda token : token.lower(), tokens)))
            .apply(lambda word_list : list(filter(lambda word: word not in stop_words, word_list)))
            .apply(lambda valid_word_list : len(valid_word_list))
          )


def _generate_stop_words(df):
    logger.info('Agregando columnas de palabras clave')
    
    df['n_tokens_title'] = tokenize_column(df, 'title')
    df['n_tokens_body'] = tokenize_column(df, 'body')

    return df


def _remove_duplicated_entries(df, column_name):
    logger.info('Removiendo entradas duplicadas')

    df.drop_duplicates(subset=[column_name], keep='first', inplace=True)

    return df


def _drop_rows_with_missing_values(df):
    logger.info('Eliminando filas con datos nulos')

    return df.dropna()


def _save_data(df, filename):
    clean_filename = f'clean_{filename}'
    logger.info(f'Guardando los archivos en: {clean_filename}')

    df.to_csv(clean_filename, encoding='utf-8-sig')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help = 'The path to the dirty data',
                        type=str)
    args = parser.parse_args()   
    
    df = main(args.filename) 
    print(df)