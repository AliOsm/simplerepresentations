from setuptools import find_packages, setup


setup(
    name='simplerepresentations',
    version='0.0.4',
    author='Ali Fadel',
    author_email='aliosm1997@gmail.com',
    description='Easy-to-use text representations extraction library based on the Transformers library.',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    keywords=['NLP', 'natural language processing', 'deep learning', 'pytorch', 'transformer', 'BERT', 'XLM', 'XLNet', 'RoBERTa', 'DistilBERT', 'GPT', 'GPT-2', 'ALBERT', 'google', 'openai', 'CMU'],
    license='Apache',
    url='https://github.com/AliOsm/simplerepresentations',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'transformers',
        'tqdm'
    ],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
