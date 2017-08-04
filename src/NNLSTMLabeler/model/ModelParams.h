#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet featAlpha;
	LSTM1Params lstm_left_param;
	LSTM1Params lstm_right_param;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		lstm_left_param.initial(opts.hiddenSize, opts.windowOutput, mem);
		lstm_right_param.initial(opts.hiddenSize, opts.windowOutput, mem);
		opts.inputSize = opts.hiddenSize * 3 * 2;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}

	bool TestInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.hiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		lstm_left_param.exportAdaParams(ada);
		lstm_right_param.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
	}

};

#endif /* SRC_ModelParams_H_ */