import shutil
from typing import Any, Tuple

import pandas as pd
import os
import tempfile
from tuhlbox.contributors import ConstantContributor, LanguageDetectionContributor


def create_df() -> Tuple[str, pd.DataFrame]:
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "text_raw"))
    rows = []

    def add_file(content: str, name: Any) -> None:
        text_path = os.path.join("text_raw", str(name))
        full_path = os.path.join(tmpdir, text_path)
        with open(full_path, "w") as o_f:
            o_f.write(content)
            rows.append(text_path)

    add_file(content="This is an English sentence.", name="1.txt")
    add_file(content="Dies ist ein Deutscher Satz.", name="2.txt")
    add_file(content="Dit is een nederlandse zin.", name="3.txt")
    return tmpdir, pd.DataFrame(dict(text_raw=rows))


def test_constant_contributor() -> None:
    tmpdir, df = create_df()
    contributor = ConstantContributor(column_name="column_name", value="value")
    actual = contributor.calculate(df)["column_name"].values.tolist()
    expected = ["value"] * 3
    assert actual == expected
    shutil.rmtree(tmpdir)


def test_language_detection_contributor() -> None:
    tmpdir, df = create_df()
    contributor = LanguageDetectionContributor()
    contributor.base_dir = tmpdir
    actual = contributor.calculate(df)["language"].values.tolist()
    expected = ["en", "de", "nl"]
    assert actual == expected
    shutil.rmtree(tmpdir)
