from setuptools import setup, find_packages

setup(
		name='RedMap',
		version='0.0.1',
		author="Charlie Hack",
		author_email="charlie@205consulting.com",
		description="a utility for scraping collections of subreddits, doing NLP on them, and visualizing.",
		packages=find_packages(),
		include_package_data=True,
		install_requires=[
			'numpy',
			'pandas',
			'gensim',
			'praw',
			'corenlp',
			'nltk',
			'unittest'
		]
)

