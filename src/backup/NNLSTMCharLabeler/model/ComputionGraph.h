#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
public:
	const static int max_sentence_length = 1024;
	const static int max_char_length = 32;

public:
	// node instances

	vector<vector<LookupNode> > _char_inputs;
	vector<WindowBuilder> _char_windows;
	vector<vector<UniNode> > _char_hiddens;

	vector<AvgPoolNode> _char_avg_poolings;
	vector<MaxPoolNode> _char_max_poolings;
	vector<MinPoolNode> _char_min_poolings;
	vector<ConcatNode> _char_concats;

	vector<LookupNode> _word_inputs;
	vector<ConcatNode> _word_represents;
	WindowBuilder _word_window;

	LSTM1Builder _lstm_left;
	LSTM1Builder _lstm_right;

	vector<ConcatNode> _lstm_concat;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;

	LinearNode _neural_output;

  Graph *_pcg;
	unordered_map<string, int>* p_word_stats;
public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int char_length){
		_char_inputs.resize(sent_length);
		_char_hiddens.resize(sent_length);
		_char_windows.resize(sent_length);
		_char_avg_poolings.resize(sent_length);
		_char_max_poolings.resize(sent_length);
		_char_min_poolings.resize(sent_length);
		_char_concats.resize(sent_length);

		for (int idx = 0; idx < sent_length; idx++) {
			_char_inputs[idx].resize(char_length);
			_char_hiddens[idx].resize(char_length);
			_char_windows[idx].resize(char_length);
			_char_avg_poolings[idx].setParam(char_length);
			_char_max_poolings[idx].setParam(char_length);
			_char_min_poolings[idx].setParam(char_length);
		}

		_word_inputs.resize(sent_length);
		_word_represents.resize(sent_length);
		_word_window.resize(sent_length);
		_lstm_left.resize(sent_length);
		_lstm_right.resize(sent_length);

		_lstm_concat.resize(sent_length);

		_avg_pooling.setParam(sent_length);
		_max_pooling.setParam(sent_length);
		_min_pooling.setParam(sent_length);
	}

	inline void clear(){
		clearVec(_char_inputs);
		_char_windows.clear();
		_char_avg_poolings.clear();
		_char_max_poolings.clear();
		_char_min_poolings.clear();
		_char_concats.clear();

		_word_inputs.clear();
		_word_represents.clear();
		_word_window.clear();
		_lstm_left.clear();
		_lstm_right.clear();
		_lstm_concat.clear();
	}

public:
	inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
    _pcg = pcg;
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
			_lstm_concat[idx].init(opts.hiddenSize * 2, -1, mem);

			for (int idy = 0; idy < _char_inputs[idx].size(); idy++) {
				_char_inputs[idx][idy].setParam(&model.chars);
				_char_inputs[idx][idy].init(opts.charDim, opts.dropProb, mem);

				_char_hiddens[idx][idy].setParam(&model.char_linear);
				_char_hiddens[idx][idy].init(opts.charHiddenSize, opts.dropProb, mem);
			}

			_char_windows[idx].init(opts.charDim, opts.charContext, mem);

			_char_avg_poolings[idx].init(opts.charHiddenSize, -1, mem);
			_char_max_poolings[idx].init(opts.charHiddenSize, -1, mem);
			_char_min_poolings[idx].init(opts.charHiddenSize, -1, mem);
			_char_concats[idx].init(opts.charHiddenSize * 3, -1, mem);
			_word_represents[idx].init(opts.wordDim + opts.charHiddenSize, -1, mem);
		}
		_word_window.init(opts.wordDim + opts.charHiddenSize, opts.wordContext, mem);
		_lstm_left.init(&model.lstm_left_param, opts.dropProb, true, mem);
		_lstm_right.init(&model.lstm_right_param, opts.dropProb, false, mem);
		_avg_pooling.init(opts.hiddenSize * 2, -1, mem);
		_max_pooling.init(opts.hiddenSize * 2, -1, mem);
		_min_pooling.init(opts.hiddenSize * 2, -1, mem);
		_concat.init(opts.hiddenSize * 3 * 2, -1, mem);
		_neural_output.setParam(&model.olayer_linear);
		_neural_output.init(opts.labelSize, -1, mem);

		p_word_stats = opts.hyper_word_stats;
	}

	string p_change_word(const string& word) {
		double p = 0.7;
		unordered_map<string, int>::iterator it;
		it = p_word_stats->find(word);
		if (it != p_word_stats->end() && it->second == 1)
		{
			double x = rand() / double(RAND_MAX);
			if (x > p)
				return unknownkey;
			else
				return word;
		}
		else
			return word;
	}

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
    _pcg->train = bTrain;
		// second step: build graph
		//forward
		int words_num = feature.m_words.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			string word;
			if (bTrain)
				word = p_change_word(feature.m_words[i]);
			else
				word = feature.m_words[i];
			_word_inputs[i].forward(_pcg, word);

			const vector<string>& chars = feature.m_chars[i];
			int char_num = chars.size();
			if (char_num > max_char_length)
				char_num = max_char_length;
			for (int j = 0; j < char_num; j++)
				_char_inputs[i][j].forward(_pcg, chars[j]);
			_char_windows[i].forward(_pcg, getPNodes(_char_inputs[i], char_num));
			for (int j = 0; j < char_num; j++)
				_char_hiddens[i][j].forward(_pcg, &_char_windows[i]._outputs[j]);
			_char_max_poolings[i].forward(_pcg, getPNodes(_char_hiddens[i], char_num));
			_word_represents[i].forward(_pcg, &_word_inputs[i], &_char_max_poolings[i]);
		}

		_word_window.forward(_pcg, getPNodes(_word_represents, words_num));

		_lstm_left.forward(_pcg, getPNodes(_word_window._outputs, words_num));
		_lstm_right.forward(_pcg, getPNodes(_word_window._outputs, words_num));

		for (int i = 0; i < words_num; i++) {
			_lstm_concat[i].forward(_pcg, &_lstm_left._hiddens[i], &_lstm_right._hiddens[i]);
		}

		_avg_pooling.forward(_pcg, getPNodes(_lstm_concat, words_num));
		_max_pooling.forward(_pcg, getPNodes(_lstm_concat, words_num));
		_min_pooling.forward(_pcg, getPNodes(_lstm_concat, words_num));
		_concat.forward(_pcg, &_avg_pooling, &_max_pooling, &_min_pooling);
		_neural_output.forward(_pcg, &_concat);
		
	}
};

#endif /* SRC_ComputionGraph_H_ */