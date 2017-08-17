#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		string strLine1;
		if (!my_getline(m_inf, strLine1))
			return NULL;
		if (strLine1.empty())
			return NULL;


		vector<string> vecInfo;
		split_bychars(strLine1, vecInfo, "||| ");
		m_instance.m_label = vecInfo[vecInfo.size() - 1];

		split_bychar(vecInfo[0], m_instance.m_words, ' ');
		m_instance.m_words.resize(vecInfo.size() - 1);
		int word_size = vecInfo.size() - 1;
		for (int idx = 0; idx < word_size; idx++)
			m_instance.m_words[idx] = normalize_to_lowerwithdigit(vecInfo[idx]);

		int word_num = m_instance.m_words.size();
		m_instance.m_chars.resize(word_num);
		for (int idx = 0; idx < word_num; idx++) {
			getCharactersFromString(m_instance.m_words[idx], m_instance.m_chars[idx]);
		}
		return &m_instance;
	}
};

#endif

