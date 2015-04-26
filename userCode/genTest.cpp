#include <string>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc < 2) return 0;
  std::ifstream fbankin("data/fbank/train.ark");
  std::ifstream labelin("data/label/train.lab");
  std::ofstream fbankout("data/fbank/trainToy.ark");
  std::ofstream labelout("data/label/trainToy.lab");

  std::string fbank, label, pre_seq = "", now_seq;
  int cnt = std::stoi(std::string(argv[1]));

  while (true)
  {
    getline(fbankin, fbank);
    getline(labelin, label);
    if (fbankin.eof()) break;
    size_t pre = 0, next;
    next = fbank.find('_');
    pre = next+1;
    next = fbank.find('_', pre);
    now_seq = fbank.substr(pre, next-pre);
    if (now_seq != pre_seq)
    {
      cnt --;
      if (cnt == -1) break;
      pre_seq = now_seq;
    }
    fbankout << fbank << std::endl;
    labelout << label << std::endl;
  }
  return 0;
}

