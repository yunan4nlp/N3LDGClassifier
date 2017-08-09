#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
public:
	const static int max_sentence_length = 1024;

public:
	// node instances
	vector<LookupNode> _word_inputs;
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
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_lstm_left.resize(sent_length);
		_lstm_right.resize(sent_length);

		_lstm_concat.resize(sent_length);

		_avg_pooling.setParam(sent_length);
		_max_pooling.setParam(sent_length);
		_min_pooling.setParam(sent_length);
	}

	inline void clear(){
		_word_inputs.clear();
		_word_window.clear();
		_lstm_left.clear();
		_lstm_right.clear();
	}

public:
	inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
    _pcg = pcg;
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb,mem);
			_lstm_concat[idx].init(opts.hiddenSize * 2, -1, mem);
		}
		_word_window.init(opts.wordDim, opts.wordContext, mem);
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
		}
		_word_window.forward(_pcg, getPNodes(_word_inputs, words_num));

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