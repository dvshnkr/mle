Answers to [2022 hw week 5](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2022/05-deployment/homework.md)

# Section 1

## Question 1

* Install Pipenv
* What's the version of pipenv you installed?
* Use `--version` to find out

**Answer**:  2023.8.23
<br>

## Question 2

* Use Pipenv to install Scikit-Learn version 1.0.2
* What's the first hash for scikit-learn you get in Pipfile.lock?

Note: you should create an empty folder for homework
and do it there.

**Answer**: sha = `08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b`
<br>

## Question 3

We've prepared a dictionary vectorizer and a model.

* Write a script for loading these models with pickle
* Score this client:

```json
{"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
```

What's the probability that this client will get a credit card?

**Answer**: 0.162
<br>

## Question 4

Now let's serve this model as a web service

* Install Flask and gunicorn (or waitress, if you're on Windows)
* Write Flask code for serving the model
* Now score this client using `requests`

```python
url = "YOUR_URL"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
```

What's the probability that this client will get a credit card?

**Answer**: 0.928
<br>

## Question 5

Download the base image `svizor/zoomcamp-model:3.9.12-slim`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

So what's the size of this base image?

**Answer**: 125MB
<br>

## Question 6

Let's run your docker container!
After running it, score this client once again:
```python
url = "YOUR_URL"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
```
What's the probability that this client will get a credit card now?

**Answer**: 0.769
<br>
