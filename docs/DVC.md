# DVC Recommendation

DVC can be useful, but it depends on your goal.

## Is DVC Worth It Here?

For this project, the CSV files are small, so DVC is not technically necessary.
Keeping the CSV files in Git is acceptable for a class or portfolio project.

However, DVC is a strong portfolio signal if you want the project to look more
like a real ML engineering workflow. It shows that you understand data
versioning, reproducibility, and separating code from larger data artifacts.

## When DVC Is A Good Idea

Use DVC if:

- the dataset becomes large
- you want to track multiple versions of the data
- you want to store data outside GitHub
- you want reproducible pipelines for training and evaluation
- you want to demonstrate ML engineering maturity

## Recommended DVC Remote Options

For a simple separate storage server:

- Google Drive
- AWS S3
- Azure Blob Storage
- Google Cloud Storage
- SSH server

For a student portfolio, Google Drive or S3 is usually enough.

## Basic Setup

Install DVC:

```powershell
pip install dvc
```

Initialize DVC:

```powershell
dvc init
```

Track data:

```powershell
dvc add data/train_energy_data.csv data/test_energy_data.csv
```

Commit the DVC metadata:

```powershell
git add data/*.dvc .gitignore .dvc
git commit -m "Track dataset with DVC"
```

Add a remote. Example for an SSH server:

```powershell
dvc remote add -d storage ssh://user@server/path/to/dvc-storage
dvc push
```

Example for S3:

```powershell
pip install "dvc[s3]"
dvc remote add -d storage s3://your-bucket/energytypenet
dvc push
```

## My Recommendation

For this exact project, DVC is optional. I would add it only if you plan to push
the repo to GitHub and want to present it as an ML engineering project.

If you mainly need a clean class project, skip DVC and keep the CSVs in `data/`.

If you want a stronger portfolio project, add DVC and mention in the README that
data is versioned separately from code.
