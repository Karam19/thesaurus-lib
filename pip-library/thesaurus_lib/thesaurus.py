import pickle
import os
import spacy
import numpy as np
from tqdm import tqdm
from gensim.utils import tokenize
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file
from bokeh.io import show, output_notebook
from collections import Counter
from .downloads import make_downloads
from nltk.stem.isri import ISRIStemmer
from sentence_transformers import SentenceTransformer
import configparser
import wget

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

output_notebook()

MAX_LENGTH = 50000000
LEMMATIZATION_THRESHOLD = 500000


class Thesaurus:
    def __init__(self, lang, user_config=None):
        self.spacy_model = None
        self.som = None
        self.fig = None
        self.model = None
        self.user_config = {}
        self.embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.lang = lang
        self.external_background = False
        if lang == 'eng' or lang == 'rus':
            config = configparser.RawConfigParser()
            conf_file = pkg_resources.read_text('thesaurus_lib', 'config.cfg')
            config.read_string(conf_file)
            self.config = dict(config.items(lang))
            self.paths = dict(config.items('paths'))
            if user_config is not None:
                config.read(user_config)
                self.user_config = dict(config.items(lang))
        else:
            raise SyntaxError("Please choose one of the following languages: ['eng', 'rus'] ")
        # download spacy model
        make_downloads(lang)
        # download pretrained model
        self.download('som_file', 'som_url')
        # download background embeddings
        self.download('back_embeds', 'embeds_url')
        self.set_spacy_model()
        self.set_som()
        if 'back_embeds' in self.user_config and 'back_tokens' in self.user_config:
            self.background_embeds, self.background_words = self.import_background(
                b_tokens=self.user_config['back_tokens'], b_embeds=self.user_config['back_embeds'])
        else:
            self.background_embeds, self.background_words = self.import_background()

    def download(self, file_key, url_key):
        if file_key in self.user_config:
            pass
        if url_key in self.user_config:
            url = self.user_config[url_key]
        else:
            url = self.config[url_key]
        wget.download(url)

    @staticmethod
    def read_text(file):
        lines = []
        for line in file:
            # line = line.decode('utf-8', 'ignore')
            lines.append(line)
        return ''.join(lines)

    def set_spacy_model(self):
        model = self.user_config['model'] if 'model' in self.user_config else self.config['model']
        self.spacy_model = spacy.load(model)
        self.spacy_model.max_length = MAX_LENGTH

    def get_nes(self, text):
        doc = self.spacy_model(text)
        nes = []
        for word in doc.ents:
            if word.label_ in ['ORG', 'GPE', 'PERSON']:
                nes.append(word.text)
        return list(dict.fromkeys(nes))

    def lemmatize(self, text, length):
        if self.lang == 'ara':
            st = ISRIStemmer()
            doc = self.spacy_model(text)
            result = " ".join([st.stem(str(token)) for token in doc])
            return result
        elif length < LEMMATIZATION_THRESHOLD:
            doc = self.spacy_model(text)
            result = " ".join([token.lemma_ for token in doc])
            return result
        else:
            for doc in self.spacy_model.pipe([text], batch_size=32, n_process=3, disable=["parser", "ner"]):
                result = " ".join([token.lemma_ for token in doc])
                return result

    @staticmethod
    def tokenize(text):
        tokens = list(tokenize(text, to_lower=True))
        return tokens

    @staticmethod
    def get_stopwords(path_, file_):
        stopwords_file = pkg_resources.read_text(path_, file_)
        stopwords = []
        for line in stopwords_file:
            stopwords.append(line[:-1])
        return stopwords

    def remove_stopwords(self, tokens: list):
        stopwords = self.get_stopwords(self.paths['stopwords_path'], self.config['stopwords_file'])
        filtered_tokens = []
        for token in tokens:
            if token not in stopwords:
                filtered_tokens.append(token)
        return filtered_tokens, list(dict.fromkeys(filtered_tokens))

    @staticmethod
    def preprocess(self, tokens):
        result = []

        for token in tokens:
            if (not token.isalpha()) or (len(token) <= 2):
                continue
            else:
                result.append(token)

        return result

    def make_embeddings(self, tokens: list) -> list:
        embeddings_filename = self.config['embeddings_file']
        if os.path.exists(embeddings_filename):
            # print('Found cache..')
            embeddings_file = open(embeddings_filename, 'rb')
            changed = False
            dictionary = pickle.load(embeddings_file)
            result = []
            for token in tokens:
                if token in dictionary:
                    result.append(dictionary[token])
                else:
                    e = self.embed_model.encode(token)
                    # e = self.spacy_model(token).vector
                    dictionary[token] = e
                    changed = True
                    result.append(e)
            if changed:
                # print('Rewriting cache..')
                embeddings_file.close()
                os.remove(embeddings_filename)
                new_embeddings_file = open(embeddings_filename, 'wb')
                pickle.dump(dictionary, new_embeddings_file)
            return result
        else:
            # print('Cache not found..')
            dictionary = dict()
            for token in tokens:
                dictionary[token] = self.embed_model.encode(token)
                # dictionary[token] = self.spacy_model(token).vector
            embeddings_file = open(embeddings_filename, 'wb')
            pickle.dump(dictionary, embeddings_file)
            return list(dictionary.values())

    @staticmethod
    def get_grid_size(n):
        neurons_num = 5 * np.sqrt(n)
        return int(np.ceil(np.sqrt(neurons_num)))

    def set_som(self):
        if 'som_file' not in self.user_config:
            model = open(self.config['som_file'], 'rb')
            som = pickle.load(model)
        else:
            som = open(self.user_config['som_file'], 'rb')
            som = pickle.load(som)
        self.model = som

    def plot_bokeh(self, background_embeds, background_words, foreground_names, preprocessed_foregrounds,
                   background_color='#d2e4f5', foreground_colors=None):

        """
        foreground_names ['foreground_name1', ...]
        preprocessed_foregrounds: {'foreground_name1': {'embeds': [...], 'words': [...]}, ...]
        """
        if foreground_colors is None:
            foreground_colors = ['#f5a09a', 'green', '#f5b19a', '#f5d59a', '#ebe428',
                                 '#28ebd1', '#1996b3', '#0b2575', '#2d0a5e', '#4d0545']

        hexagon_size = 10
        dot_size = 4
        grid_size = 100

        plot_size = int(hexagon_size * grid_size * 1.5)
        # print(plot_size)

        som = self.model
        if os.path.isfile(self.config['index_file']) or self.external_background is False:
            if self.external_background is False:
                index = pkg_resources.read_binary(self.paths['index_path'], self.config['index_file'])
                index = pickle.loads(index)
            else:
                with open(self.config['index_file'], 'rb') as index_file:
                    index = pickle.load(index_file)

            b_label = []

            b_weight_x, b_weight_y = [], []
            for cnt, i in enumerate(background_embeds):
                w = index[background_words[cnt]]

                wx, wy = som.convert_map_to_euclidean(xy=w)
                wy = wy * np.sqrt(3) / 2
                b_weight_x.append(wx)
                b_weight_y.append(wy)
                b_label.append(background_words[cnt])

        else:
            index = dict()

            b_label = []

            b_weight_x, b_weight_y = [], []
            for cnt, i in enumerate(background_embeds):
                w = som.winner(i)
                index[background_words[cnt]] = w
                wx, wy = som.convert_map_to_euclidean(xy=w)
                wy = wy * np.sqrt(3) / 2
                b_weight_x.append(wx)
                b_weight_y.append(wy)
                b_label.append(background_words[cnt])

            with open(self.config['index_file'], 'wb') as index_file:
                pickle.dump(index, index_file)

        # translations = [(-0.15, -0.15), (0.15, 0.15), (-0.15, 0.15)]
        translations = [(-0.15, -0.15), (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15), (0.15, 0.15),
                        (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15), (0.15, 0.15)]

        for foreground_unit in foreground_names:
            label = []
            weight_x, weight_y = [], []

            fu = preprocessed_foregrounds[foreground_unit]

            for cnt, i in enumerate(fu['embeds']):
                if fu['words'][cnt] in index:
                    w = index[fu['words'][cnt]]
                else:
                    w = som.winner(i)
                wx, wy = som.convert_map_to_euclidean(xy=w)
                wy = wy * np.sqrt(3) / 2
                weight_x.append(wx)
                weight_y.append(wy)
                label.append(fu['words'][cnt])

            fu['label'] = label
            fu['weight_x'] = weight_x
            fu['weight_y'] = weight_y

        output_file("som_" + self.lang + ".html")
        fig = figure(plot_height=plot_size, plot_width=plot_size,
                     match_aspect=True,
                     tools="pan, wheel_zoom, reset, save")

        fig.axis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None

        # create data stream for plotting
        b_source_pages = ColumnDataSource(
            data=dict(
                wx=b_weight_x,
                wy=b_weight_y,
                species=b_label
            )
        )

        all_weights = []
        for foreground_unit in foreground_names:
            fu = preprocessed_foregrounds[foreground_unit]
            temp = [(fu['weight_x'][i], fu['weight_y'][i]) for i in range(len(fu['weight_x']))]
            all_weights += temp

            temp_c = Counter(temp)
            fu['local_counts'] = temp_c

        all_weights_count = Counter(all_weights)

        for foreground_unit in foreground_names:
            fu = preprocessed_foregrounds[foreground_unit]

            translation = translations.pop(0)
            translations.append(translation)

            hex_ = {'weight_x': [], 'weight_y': [], 'label': [], 'size': []}
            for i in range(len(fu['weight_x'])):
                coords = (fu['weight_x'][i], fu['weight_y'][i])
                if all_weights_count[coords] - fu['local_counts'][coords] > 0:
                    hex_['weight_x'].append(fu['weight_x'][i] + translation[0])
                    hex_['weight_y'].append(fu['weight_y'][i] + translation[1])
                    hex_['size'].append(dot_size)
                else:
                    hex_['weight_x'].append(fu['weight_x'][i])
                    hex_['weight_y'].append(fu['weight_y'][i])
                    hex_['size'].append(hexagon_size)
                hex_['label'].append(fu['label'][i])

            hex_pages = ColumnDataSource(
                data=dict(
                    wx=hex_['weight_x'],
                    wy=hex_['weight_y'],
                    species=hex_['label'],
                    size=hex_['size']
                )
            )
            fu['hex_pages'] = hex_pages

        fig.hex(x='wy', y='wx', source=b_source_pages,
                fill_alpha=0.2, fill_color=background_color,
                line_alpha=1.0, line_color=background_color, line_width=1,
                size=hexagon_size, name="one",
                legend_label='Background')

        for foreground_unit in foreground_names:
            fu = preprocessed_foregrounds[foreground_unit]
            current_color = foreground_colors.pop(0)
            foreground_colors.append(current_color)
            fig.hex(x='wy', y='wx', source=fu['hex_pages'],
                    fill_color=current_color,
                    line_width=0.1,
                    size='size', name="two",
                    legend_label=foreground_unit)

        fig.legend.location = "top_left"
        fig.add_layout(fig.legend[0], 'right')
        fig.legend.click_policy = "hide"
        fig.add_tools(HoverTool(
            tooltips=[
                ("label", '@species')],
            mode="mouse",
            attachment='above',
            point_policy="follow_mouse",
            names=["one", "two", "three"]
        ))

        return fig, som

    def process_texts(self, texts):

        # bert = Bert()

        all_embeddings = []
        all_words = []

        for source_text in tqdm(texts):

            lemmatized_text = self.lemmatize(source_text, len(source_text))

            tokenized_text = self.tokenize(lemmatized_text)

            filtered_tokens, filtered_tokens_set = self.remove_stopwords(tokenized_text)

            new = []
            cb = Counter(filtered_tokens)
            occurrence = 1
            for tok in filtered_tokens_set:
                if cb[tok] == occurrence:
                    new.append(tok)

            processed_tokens_set = self.preprocess(self, new)

            embeddings = self.make_embeddings(processed_tokens_set)

            oov_words = []

            for i in range(len(processed_tokens_set)):
                if processed_tokens_set[i] not in all_words:
                    if np.any(embeddings[i]):
                        all_embeddings.append(embeddings[i])
                        all_words.append(processed_tokens_set[i])
                    else:
                        oov_words.append(processed_tokens_set[i])
                        # all_embeddings.append(bert.bert_embedding(processed_tokens_set[i]))
                        # all_words.append(processed_tokens_set[i])

            # print(oov_words)

        return all_embeddings, all_words

    @staticmethod
    def read_txt(file_name):
        with open(file_name, 'r') as file:
            return file.read()

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def custom_preprocessing_of_data(data):
        res = []
        num_of_articles = 10
        for article in data[:num_of_articles]:
            try:
                res.append(article['clean'])
            except KeyError:
                continue

        return res

    def process_foreground(self, foreground_names, texts):
        processed_foregrounds = dict()

        for foreground_unit in tqdm(foreground_names):
            all_embeddings_of_unit, all_words_of_unit = self.process_texts(texts[foreground_unit])

            one_processed_foreground = {'embeds': all_embeddings_of_unit, 'words': all_words_of_unit}
            processed_foregrounds[foreground_unit] = one_processed_foreground

        return processed_foregrounds

    def import_background(self, b_tokens=None, b_embeds=None):
        background_embeds, background_words = None, None
        if b_tokens is None and b_embeds is None:
            background_words = pkg_resources.read_binary(self.paths['back_tokens_path'], self.config['back_tokens'])
            background_embeds = open(self.config['back_embeds'], 'rb')
            background_words, background_embeds = pickle.loads(background_words), pickle.load(background_embeds)
        else:
            tokens = open(b_tokens, 'rb')
            if b_tokens.lower().endswith('.pickle'):
                background_words = pickle.load(tokens)
            embeds = open(b_embeds, 'rb')
            background_embeds = pickle.load(embeds)
            self.external_background = True

        return background_embeds, background_words

    def show_map(self, foreground_names, processed_foregrounds):
        fig, som = self.plot_bokeh(self.background_embeds, self.background_words, foreground_names, processed_foregrounds, )
        self.som = som
        self.fig = fig
        show(fig)

    def search(self, words, embeds, search_word, search_color='blue'):

        som = self.som
        fig = self.fig
        try:
            index = words.index(search_word)

            label = []

            weight_x, weight_y = [], []

            w = som.winner(embeds[index])
            wx, wy = som.convert_map_to_euclidean(xy=w)
            wy = wy * np.sqrt(3) / 2
            weight_x.append(wx)
            weight_y.append(wy)
            label.append(search_word)

            source_pages = ColumnDataSource(
                data=dict(
                    wx=weight_x,
                    wy=weight_y,
                    species=label
                )
            )

            point = fig.scatter(x='wy', y='wx', source=source_pages,
                                line_width=0.1, fill_color=search_color, size=4)
            circle = fig.scatter(x='wy', y='wx', source=source_pages,
                                 line_color=search_color, line_width=1, line_alpha=1,
                                 fill_alpha=0,
                                 size=160)

            show(fig)

            return point, circle

        except ValueError:
            print('No such a word in map')

            return None, None
