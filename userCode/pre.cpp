#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

class Word
{
  public:
  std::string speaker_id;
  std::string sequence_id;
  int word_id;
  std::string vec;

  bool operator<(const Word &t) const
  {
    if (speaker_id != t.speaker_id)
      return speaker_id < t.speaker_id;
    if (sequence_id != t.sequence_id)
      return sequence_id < t.sequence_id;
    return word_id < t.word_id;
  }
};

int main()
{
  std::ifstream fin("../data/label/train.lab");
  std::ofstream fout("../data/label/test.lab");
  std::string line;
  std::vector<Word> words;
  while(getline(fin, line))
  {
    Word now;
    if (fin.eof()) break;
    size_t pre = 0, next;
    next = line.find('_');
    now.speaker_id = line.substr(pre, next-pre);
    pre = next+1;

    next = line.find('_', pre);
    now.sequence_id = line.substr(pre, next-pre);
    pre = next+1;

    next = line.find(',', pre);
    now.word_id = std::stoi(line.substr(pre, next-pre));
    pre = next+1;
    
    now.vec = line.substr(pre);
    
    words.push_back(now);
  }
  std::sort(words.begin(), words.end());
  for (int i=0; i<words.size(); i++)
  {
    fout << words[i].speaker_id+"_"+words[i].sequence_id+"_"+std::to_string(words[i].word_id)+","+words[i].vec << std::endl;
  }
  std::cout << words.size() << std::endl;
}

