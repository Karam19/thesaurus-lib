#############
Thesaurus library
#############
*************
How to run
*************
Minimalistic way
=============
Install the library:

..  code-block:: bash
    :caption: in your terminal
    pip install thesaurus-lib
Create an object and specify the language:

..  code-block:: python
    :caption: in your terminal
    obj = Thesaurus(lang='eng')
Show output:

..  code-block:: python
    obj.show_map()
Run with your own foregrounds:
=============
After you install the library and create the object do the following

1. pass them to the library:

..  code-block:: python
    text1 = obj.read_pickle('2017')
    text2 = obj.read_txt('shakespeare.txt')
    text3 = obj.read_text('My foreground in string format')
2. Preprocess your foreground:

..  code-block:: python
    texts = dict()
    foreground_name = 'Physics articles 2017'
    texts[foreground_name] = obj.custom_preprocessing_of_data(text1)
3. Process foregrounds:

..  code-block:: python
    processed_foregrounds = obj.process_foreground(foreground_names, texts)
4. Show output:

..  code-block:: python
    obj.show_map()
Use your own configurations
=============
After installing the library create a file called 'config.cfg' in your working
directory and fill the value with your own files:

..  code-block:: cfg
    [paths]
    som_path =
    index_path =
    back_tokens_path =
    back_embeds_path =
    stopwords_path =
    foregrounds_path =

    [lang]
    som_url =
    embeds_url =
    som_file =
    index_file =
    back_tokens =
    back_embeds =
    embeddings_file =
    STOPWORDS_FILE =
    model =

Note:
Don't leave any empty field in config.cfg. For example if you aren't providing a som_file
then delete it in your config.cfg and don't keep it in this way:

..  code-block:: cfg
    # fill it or delete it
    som_file =

