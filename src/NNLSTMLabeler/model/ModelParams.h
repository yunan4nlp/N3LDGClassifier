#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet featAlpha;
	LSTM1Params lstm_param;
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
		lstm_param.initial(opts.hiddenSize, opts.windowOutput, mem);
		opts.inputSize = opts.hiddenSize * 3;
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
		lstm_param.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words E");
		checkgrad.add(&lstm_param.input.W1, "lstm_param.input.W1 W");
		checkgrad.add(&lstm_param.cell.W1, "lstm_param.cell.W1 W");
		checkgrad.add(&lstm_param.output.W1, "lstm_param.output.W1 W");

		checkgrad.add(&lstm_param.input.W2, "lstm_param.input.W2 W");
		checkgrad.add(&lstm_param.cell.W2, "lstm_param.cell.W2 W");
		checkgrad.add(&lstm_param.output.W2, "lstm_param.output.W2 W");

		checkgrad.add(&olayer_linear.W, "output layer W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
		wordAlpha.write(os);
		words.save(os);
		lstm_param.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		lstm_param.load(is, mem);
		olayer_linear.load(is, mem);
		labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */