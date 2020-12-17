"""Dataset manipulation scripts."""

import json
import pickle
import shutil
from glob import glob

import click
import os
import warnings
import pandas as pd
import stanza
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

FEATURE_TEXT = 'text_raw'
FEATURE_STANZA = 'stanza'
FEATURE_CONSTITUENTS = 'constituencies'
LANGUAGE_COLUMN = 'language'
DATASET_CSV = 'dataset.csv'


@click.command(help='transforms corpora created with the reddit script into '
                    'the common dataframe format.')
@click.argument('input_directory')
def reddit_to_common(input_directory):
    """
    Transform corpora created with reddit scripts into dataframe format.

    This script is meant to be called via CLI. It will read a directory that
    contains a nested directory structure and produces a common dataset format
    using DataFrames. The result is a meta-information file called dataset.csv
    and a directory 'text_raw' which contains the raw text files.

    Args:
        input_directory: The directory that contains the topmost target
            directories (often, authors).

    Returns: Nothing.

    """
    records = []

    text_directory = os.path.join(input_directory, FEATURE_TEXT)
    if not os.path.isdir(text_directory):
        os.makedirs(text_directory)

    old_directory = os.path.join(input_directory, 'old')
    if not os.path.isdir(old_directory):
        os.makedirs(old_directory)

    meta_file = os.path.join(input_directory, DATASET_CSV)

    authors = sorted(os.listdir(input_directory))
    for author in tqdm(authors):

        # don't process these "authors" that might be existing from previous
        # calls to this script:
        if author in [os.path.basename(text_directory),
                      os.path.basename(meta_file)]:
            continue

        author_dir = os.path.join(input_directory, author)
        if not os.path.isdir(author_dir):
            logger.warning('not an author directory: %s', author_dir)
            continue
        languages = os.listdir(author_dir)
        for language in languages:
            language_dir = os.path.join(author_dir, language)
            if not os.path.isdir(language_dir):
                logger.warning('not a language directory: %s', language_dir)
                continue
            json_files = glob(language_dir + '/*.json')
            for json_file in json_files:
                name_ext = os.path.basename(json_file)
                name = os.path.splitext(name_ext)[0]
                text_name = f'{author}_{language}_{name}.txt'
                text_file = os.path.join(FEATURE_TEXT, text_name)
                full_text_file = os.path.join(input_directory, text_file)
                with open(json_file) as i_f, open(full_text_file, 'w') as o_f:
                    js = json.load(i_f)
                    o_f.write(js['body_clean'])
                    del js['body']
                    del js['body_clean']
                    js[FEATURE_TEXT] = text_file
                    js['group_field'] = language  # important for deepl_...
                    records.append(js)

        # move old author dir
        shutil.move(author_dir, old_directory)

    df = pd.DataFrame.from_records(records)
    df.to_csv(meta_file, index=False)


@click.command(help='Reads dataset.csv + column name to produce stanza dir')
@click.argument('input-directory')
@click.option('-t', '--text-column-name', default=FEATURE_TEXT)
@click.option('-l', '--language-column-name', default=LANGUAGE_COLUMN)
@click.option('-o', '--overwrite', default=False)
@click.option('-out', '--output-column-name', default=FEATURE_STANZA)
def parse_dependency(input_directory, text_column_name, language_column_name,
                     overwrite, output_column_name):
    """
    Parse text files using the stanza parser.

    Args:
        input_directory: Directory containing the dataset.csv file, and a
            directory named <text_column_name>
        text_column_name: name of the column containing the raw text data.
            defaults to 'text_raw'.
        language_column_name:  name of the column where the language is saved.
            defaults to 'language'.
        overwrite: whether to re-parse existing files. If this is not set, this
            script will leave already parsed files untouched and won't parse
            the appropriate inputs.
        output_column_name: name of the output column in the meta-data file.
            defaults to 'stanza'.

    Returns: Nothing, this is a cli script.

    """
    warnings.simplefilter('ignore')  # stanza is very 'loud'.
    main_dataset_file = os.path.join(input_directory, DATASET_CSV)
    df = pd.read_csv(main_dataset_file)

    # check output directory
    out_dir = os.path.join(input_directory, output_column_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    def get_filename(text_filename):
        name = os.path.splitext(os.path.basename(text_filename))[0] + '.pckl'
        return os.path.join(output_column_name, name)

    df[output_column_name] = df[text_column_name].apply(get_filename)
    tuples = [(in_file, out_file, language)
              for (in_file, out_file, language)
              in zip(df[text_column_name],
                     df[output_column_name],
                     df[language_column_name])
              if (not os.path.isfile(os.path.join(input_directory, out_file)))
              or overwrite]
    if not tuples:
        logger.warning('no files remaining, skipping calculation')
        return

    parsers = {}
    for in_file, out_file, language in tqdm(tuples):
        if language.endswith('_to_en'):
            language = 'en'
        if language not in parsers:
            parsers[language] = stanza.Pipeline(lang=language, use_gpu=False)
        parser = parsers[language]
        with open(os.path.join(input_directory, in_file)) as in_fh:
            parsed = parser(in_fh.read())
        with open(os.path.join(input_directory, out_file), 'wb') as out_fh:
            pickle.dump(parsed, out_fh)
    logger.info('writing %s', main_dataset_file)
    df.to_csv(main_dataset_file, index=False)


@click.command(help='reads dataset.csv, produces constituencies directory')
@click.argument('input-directory')
@click.option('-t', '--text-column-name', default=FEATURE_TEXT)
@click.option('-l', '--language-column-name', default=LANGUAGE_COLUMN)
def parse_constituency(input_directory, text_column_name,
                       language_column_name):
    """
    Ignore this method for the time being.

    This is no longer working unless the common file transformers are replaced.
    """
    pass
    # meta = os.path.join(input_directory, DATASET_CSV)
    #
    # df = pd.read_csv(meta)
    #
    # out_dir = os.path.join(input_directory, FEATURE_CONSTITUENTS)
    # if not os.path.isdir(out_dir):
    #     os.makedirs(out_dir)
    #
    # sub_dfs = {}
    #
    # columns = [text_column_name, language_column_name]
    # for language, sub_df in df[columns].groupby(language_column_name):
    #     logger.info('parsing %s', language)
    #
    #     infiles = list(sub_df[text_column_name].values)
    #     # the path to the text file points to a subfolder
    #     outfiles = PathTransformer('../' + FEATURE_CONSTITUENTS,
    #                                '.json').transform(infiles)
    #     pairs = [(i_f, o_f) for (i_f, o_f) in zip(infiles, outfiles)
    #              if not os.path.isfile(o_f)]
    #     if not pairs:
    #         logger.warning('no files remaining, skipping calculation')
    #         return
    #
    #     input_filenames, output_filenames = list(zip(*pairs))
    #
    #
    #
    #     make_pipeline(
    #         FileReader(),
    #         CoreNLPTreeTransformer(
    #             language=language,
    #             port=CORENLP_PORT,
    #             model_dir=CORENLP_MODELS,
    #             properties_dir=CORENLP_PROPERTIES),
    #         FileWriter(output_filenames, writemode='json'),
    #     ).transform(input_filenames)
    #
    #     sub_df[FEATURE_CONSTITUENTS] = output_filenames
    #     sub_dfs[language] = sub_df
    #
    # pd.concat(sub_dfs, axis=0).to_csv(DATASET_CSV + '2', index=False)
