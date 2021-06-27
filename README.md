# Petnica projekat 2021

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


## Tehnikalije
- Packaging: `pipenv` *+ `pyenv`*
- Linter: `flake8`
- Formatter: `black`


## Pokretanje projekta
Nakon kloniranja repozitorijuma:
```bash
pipenv install --dev
```
Pokretanje jupyter lab-a:
```bash
pipenv run jupyter lab
```
Konzola unutar virtualnog okruzenja:
```bash
pipenv shell
```

## Uputstva
`.py` fajlovi untar `notebooks/` foldera se ne eidtuju već se koriste kao format za verzionisanje pomoću kog se rekonstruiše `.ipynb` fajl koristeći `jupytext` ekstenziju. [Detaljnije](https://jupytext.readthedocs.io/en/latest/examples.html#collaborating-on-jupyter-notebooks)

Za pokretanje `.py` verzije jupyter sveske potrebno je cd-ovati u direktorijum u kome se nalazi zbog razrešavanja putanja lokalnih modula.

`pre-commit hook` pre svakog commit-a ceo codebase formatira pa lintuje i ako fail-uje u nekom koraku prekida commit.


## Preporučena podešavanja za VS Code
```json
// settings.json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length=79"
    ]
}
```

## Reference
[Literatura](./literatura.md)
