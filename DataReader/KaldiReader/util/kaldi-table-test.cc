// util/kaldi-table-test.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "base/io-funcs.h"
#include "util/kaldi-io.h"
#include "base/kaldi-math.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "util/table-types.h"
#ifndef _MSC_VER
#include <unistd.h> // for sleep.
#endif

namespace kaldi {

void UnitTestReadScriptFile() {
  typedef std::pair<std::string, std::string>  pr;
  {
    std::stringstream ss;
    ss << " a b\n";
    ss << "c d \n";
    ss << "c d e \n";
    std::vector<pr> script;
    bool ans = ReadScriptFile(ss, true, &script);
    KALDI_ASSERT(ans);
    std::vector<pr> script2;
    script2.push_back(std::make_pair<std::string, std::string>("a", "b"));
    script2.push_back(std::make_pair<std::string, std::string>("c", "d"));
    script2.push_back(std::make_pair<std::string, std::string>("c", "d e"));
    KALDI_ASSERT(script == script2);
  }
  {
    typedef std::pair<std::string, std::string>  pr;
    std::stringstream ss;
    ss << " a \n";
    std::vector<pr> script;
    // suppress the warning since I already checked it's OK.
    KALDI_ASSERT(!ReadScriptFile(ss, false, &script));
  }
  {
    typedef std::pair<std::string, std::string>  pr;
    std::stringstream ss;
    ss << "\n";
    std::vector<pr> script;
    // suppress the warning since I already checked it's OK.
    KALDI_ASSERT(!ReadScriptFile(ss, false, &script));
  }
#ifndef _MSC_VER
  {
    Output ko("| gzip -c > tmpf.gz", false);  // text mode.
    ko.Stream() << "a b\n";
    ko.Close();
    std::vector<pr> script;
    sleep(1);  // This test does not work without this sleep:
    bool ans = ReadScriptFile("gunzip -c tmpf.gz |", true, &script);
    KALDI_ASSERT(ans);
    std::vector<pr> script2;
    script2.push_back(std::make_pair<std::string, std::string>("a", "b"));
    KALDI_ASSERT(script == script2);
  }

  {
    Output ko("| gzip -c > tmpf.gz", true);  // binary mode w/ header: should fail,
    // because script files should not have binary header.
    ko.Stream() << "a b\n";
    bool ans = ko.Close();
    KALDI_ASSERT(ans);
    sleep(1);  // This test does not work without this sleep:
    // seems to be some kind of file-system latency.
    std::vector<pr> script;
    ans = ReadScriptFile("gunzip -c tmpf.gz |", false, &script);
    KALDI_ASSERT(!ans);
  }
#endif
}


void UnitTestClassifyWspecifier() {

  {
    std::string a = "b,ark:foo|";
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kArchiveWspecifier && ark == "foo|" && scp == "" && opts.binary == true);
  }

  {
    std::string a = "t,ark:foo|";
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kArchiveWspecifier && ark == "foo|" && scp == "" && opts.binary == false);
  }

  {
    std::string a = "t,scp:a b c d";
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kScriptWspecifier && ark == "" && scp == "a b c d" && opts.binary == false);
  }

  {
    std::string a = "t,ark,scp:a b,c,d";
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kBothWspecifier && ark == "a b" && scp == "c,d" && opts.binary == false);
  }

  {
    std::string a = "";
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kNoWspecifier);
  }

  {
    std::string a = " t,ark:boo";
    WspecifierType ans = ClassifyWspecifier(a, NULL, NULL, NULL);
    KALDI_ASSERT(ans == kNoWspecifier);
  }

  {
    std::string a = " t,ark:boo";  // leading space not allowed.
    WspecifierType ans = ClassifyWspecifier(a, NULL, NULL, NULL);
    KALDI_ASSERT(ans == kNoWspecifier);
  }

  {
    std::string a = "t,ark:boo ";  // trailing space not allowed.
    WspecifierType ans = ClassifyWspecifier(a, NULL, NULL, NULL);
    KALDI_ASSERT(ans == kNoWspecifier);
  }

  {
    std::string a = "b,ark,scp:,";  // empty ark, scp fnames valid.
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kBothWspecifier && ark == "" && scp == "" && opts.binary == true);
  }

  {
    std::string a = "f,b,ark,scp:,";  // empty ark, scp fnames valid.
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kBothWspecifier && ark == "" && scp == "" && opts.binary == true && opts.flush == true);
  }

  {
    std::string a = "nf,b,ark,scp:,";  // empty ark, scp fnames valid.
    std::string ark = "x", scp = "y"; WspecifierOptions opts;
    WspecifierType ans = ClassifyWspecifier(a, &ark, &scp, &opts);
    KALDI_ASSERT(ans == kBothWspecifier && ark == "" && scp == "" && opts.binary == true && opts.flush == false);
  }


}


void UnitTestClassifyRspecifier() {

  {
    std::string a = "ark:foo|";
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kArchiveRspecifier && fname == "foo|");
  }


  {
    std::string a = "b,ark:foo|";  // b, is ignored.
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kArchiveRspecifier && fname == "foo|");
  }

  {
    std::string a = "ark,b:foo|";  // , b is ignored.
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kArchiveRspecifier && fname == "foo|");
  }


  {
    std::string a = "scp,b:foo|";
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "foo|");
  }

  {
    std::string a = "scp,scp,b:foo|";  // invalid as repeated.
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kNoRspecifier && fname == "");
  }

  {
    std::string a = "ark,scp,b:foo|";  // invalid as combined.
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kNoRspecifier && fname == "");
  }

  {
    std::string a = "scp,o:foo|";
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "foo|");
    KALDI_ASSERT(opts.once);
  }

  {
    std::string a = "scp,no:foo|";
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "foo|");
    KALDI_ASSERT(!opts.once);
  }

  {
    std::string a = "s,scp,no:foo|";
    std::string fname = "x";
    RspecifierOptions opts;
    RspecifierType ans = ClassifyRspecifier(a, &fname, &opts);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "foo|");
    KALDI_ASSERT(!opts.once && opts.sorted);
  }


  {
    std::string a = "scp:foo|";
    std::string fname = "x";
    RspecifierType ans = ClassifyRspecifier(a, &fname, NULL);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "foo|");
  }

  {
    std::string a = "scp:";  // empty fname valid.
    std::string fname = "x";
    RspecifierType ans = ClassifyRspecifier(a, &fname, NULL);
    KALDI_ASSERT(ans == kScriptRspecifier && fname == "");
  }

  {
    std::string a = "scp:";  // empty fname valid.
    RspecifierType ans = ClassifyRspecifier(a, NULL, NULL);
    KALDI_ASSERT(ans == kScriptRspecifier);
  }

  {
    std::string a = "";
    RspecifierType ans = ClassifyRspecifier(a, NULL, NULL);
    KALDI_ASSERT(ans == kNoRspecifier);
  }

  {
    std::string a = "scp";
    RspecifierType ans = ClassifyRspecifier(a, NULL, NULL);
    KALDI_ASSERT(ans == kNoRspecifier);
  }

  {
    std::string a = "ark";
    RspecifierType ans = ClassifyRspecifier(a, NULL, NULL);
    KALDI_ASSERT(ans == kNoRspecifier);
  }

  {
    std::string a = "ark:foo ";  // trailing space not allowed.
    RspecifierType ans = ClassifyRspecifier(a, NULL, NULL);
    KALDI_ASSERT(ans == kNoRspecifier);
  }

  // Testing it accepts the meaningless t, and b, prefixes.
  {
    std::string a = "b,scp:a", b;
    RspecifierType ans = ClassifyRspecifier(a, &b, NULL);
    KALDI_ASSERT(ans == kScriptRspecifier && b == "a");
  }
  {
    std::string a = "t,scp:a", b;
    RspecifierType ans = ClassifyRspecifier(a, &b, NULL);
    KALDI_ASSERT(ans == kScriptRspecifier && b == "a");
  }
  {
    std::string a = "b,ark:a", b;
    RspecifierType ans = ClassifyRspecifier(a, &b, NULL);
    KALDI_ASSERT(ans == kArchiveRspecifier && b == "a");
  }
  {
    std::string a = "t,ark:a", b;
    RspecifierType ans = ClassifyRspecifier(a, &b, NULL);
    KALDI_ASSERT(ans == kArchiveRspecifier && b == "a");
  }


}

void UnitTestTableSequentialInt32(bool binary) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<int32> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( rand() );
  }

  bool ans;
  Int32Writer bw(binary ? "b,ark:tmpf" : "t,ark:tmpf");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialInt32Reader sbr("ark:tmpf");
  std::vector<std::string> k2;
  std::vector<int32> v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}

void UnitTestTableSequentialBool(bool binary) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<bool> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( (rand()%2 == 0) );
  }

  bool ans;
  BoolWriter bw(binary ? "b,ark:tmpf" : "t,ark:tmpf");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialBoolReader sbr("ark:tmpf");
  std::vector<std::string> k2;
  std::vector<bool> v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}


void UnitTestTableSequentialDouble(bool binary) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<double> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( (rand() / static_cast<double>(rand()) ));
  }

  bool ans;
  DoubleWriter bw(binary ? "b,ark:tmpf" : "t,ark:tmpf");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialDoubleReader sbr("ark:tmpf");
  std::vector<std::string> k2;
  std::vector<double> v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  if (binary) {
    KALDI_ASSERT(v2 == v);
  } else {
    KALDI_ASSERT(v2.size() == v.size());
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(ApproxEqual(v[i], v2[i]));
  }
}


// Writing as both and reading as archive.
void UnitTestTableSequentialDoubleBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<double> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( (rand() / static_cast<double>(rand()) ));
  }

  bool ans;
  DoubleWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialDoubleReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<double> v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  if (binary) {
    KALDI_ASSERT(v2 == v);
  } else {
    KALDI_ASSERT(v2.size() == v.size());
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(ApproxEqual(v[i], v2[i]));
  }
}


// Writing as both and reading as archive.
void UnitTestTableSequentialInt32VectorBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<std::vector<int32> > v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( std::vector<int32>() );
    int32 sz2 = rand() % 5;
    for (int32 j = 0; j  < sz2; j++)
      v.back().push_back( rand() % 100);
  }

  bool ans;
  Int32VectorWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialInt32VectorReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<std::vector<int32> > v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}


// Writing as both and reading as archive.
void UnitTestTableSequentialInt32PairVectorBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k(sz);
  std::vector<std::vector<std::pair<int32, int32> > > v(sz);

  for (int32 i = 0; i < sz; i++) {
    k[i] = CharToString( 'a' + static_cast<char>(i));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    int32 sz2 = rand() % 5;
    for (int32 j = 0; j < sz2; j++) 
      v[i].push_back(std::pair<int32, int32>(rand() % 10, rand() % 10));
  }
  
  bool ans;
  Int32PairVectorWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialInt32PairVectorReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<std::vector<std::pair<int32, int32> > > v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}


// Writing as both and reading as archive.
void UnitTestTableSequentialInt32VectorVectorBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<std::vector<std::vector<int32> > > v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( std::vector<std::vector<int32> >() );
    int32 sz2 = rand() % 5;
    for (int32 j = 0; j  < sz2; j++) {
      v.back().push_back(std::vector<int32>() );
      int32 sz3 = rand() % 2;
      for (int32 k = 0; k  < sz3; k++)
        v.back().back().push_back( rand() % 100);
    }
  }

  bool ans;
  Int32VectorVectorWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialInt32VectorVectorReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<std::vector<std::vector<int32> > > v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}


void UnitTestTableSequentialInt32Script(bool binary) {
  int32 sz = rand() % 10;
  std::vector<std::pair<std::string, std::string> > script;
  std::vector<std::string> k;
  std::vector<int32> v;

  for (int32 i = 0; i < sz; i++) {
    char buf[3];
    buf[0] = 'a' + static_cast<char>(i);
    buf[1] = (i%2 == 0 ? 'b'+static_cast<char>(i) : '\0');
    buf[2] = '\0';
    k.push_back( std::string(buf));
    script.push_back( std::make_pair(std::string(buf), std::string(buf) + ".tmp"));
    v.push_back( rand() );
  }

  WriteScriptFile("tmp.scp", script);
  {
    std::vector<std::pair<std::string, std::string> > script2;
    ReadScriptFile("tmp.scp", true, &script2);
    KALDI_ASSERT(script2 == script);  // This tests WriteScriptFile and ReadScriptFile.
  }

  bool ans;
  Int32Writer bw(binary ? "b,scp:tmp.scp" : "t,scp:tmp.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialInt32Reader sbr("scp:tmp.scp");
  std::vector<std::string> k2;
  std::vector<int32> v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(sbr.Value());
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  KALDI_ASSERT(v2 == v);
}

// Writing as both and reading as archive.
void UnitTestTableSequentialDoubleMatrixBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<Matrix<double>*> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( new Matrix<double>(1 + rand()%4, 1 + rand() % 4));
    for (int32 i = 0; i < v.back()->NumRows(); i++)
      for (int32 j = 0; j < v.back()->NumCols(); j++)
        (*(v.back()))(i, j) = RandGauss();
  }

  bool ans;
  DoubleMatrixWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], *(v[i]));
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialDoubleMatrixReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<Matrix<double>* > v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(new Matrix<double>(sbr.Value()));
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  if (binary) {
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(v2[i]->ApproxEqual(*(v[i]), 1.0e-10));
  } else {
    KALDI_ASSERT(v2.size() == v.size());
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(v2[i]->ApproxEqual(*(v[i])));
  }
  for (int32 i = 0; i < sz; i++) {
    delete v[i];
    delete v2[i];
  }
}


// Writing as both and reading as archive.
void UnitTestTableSequentialBaseFloatVectorBoth(bool binary, bool read_scp) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<Vector<BaseFloat>*> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( new Vector<BaseFloat>(1 + rand()%4));
    for (int32 i = 0; i < v.back()->Dim(); i++)
      (*(v.back()))(i) = RandGauss();
  }

  bool ans;
  BaseFloatVectorWriter bw(binary ? "b,ark,scp:tmpf,tmpf.scp" : "t,ark,scp:tmpf,tmpf.scp");
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], *(v[i]));
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);

  SequentialBaseFloatVectorReader sbr(read_scp ? "scp:tmpf.scp" : "ark:tmpf");
  std::vector<std::string> k2;
  std::vector<Vector<BaseFloat>* > v2;
  for (; !sbr.Done(); sbr.Next()) {
    k2.push_back(sbr.Key());
    v2.push_back(new Vector<BaseFloat>(sbr.Value()));
  }
  KALDI_ASSERT(sbr.Close());
  KALDI_ASSERT(k2 == k);
  if (binary) {
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(v2[i]->ApproxEqual(*(v[i]), 1.0e-10));
  } else {
    KALDI_ASSERT(v2.size() == v.size());
    for (size_t i = 0; i < v2.size(); i++)
      KALDI_ASSERT(v2[i]->ApproxEqual(*(v[i])));
  }
  for (int32 i = 0; i < sz; i++) {
    delete v[i];
    delete v2[i];
  }
}

template<class T> void RandomizeVector(std::vector<T> *v) {
  if (v->size() > 1) {
    for (size_t i = 0; i < 10; i++) {
      size_t j = rand() % v->size(),
          k = rand() % v->size();
      if (j != k)
        std::swap((*v)[j], (*v)[k]);
    }
  }
}


// Writing as both scp and archive, with random access.

void UnitTestTableRandomBothDouble(bool binary, bool read_scp,
                                    bool sorted, bool called_sorted,
                                    bool once) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<double> v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.push_back( (rand() / static_cast<double>(rand()) ));
  }

  if (!sorted)
    RandomizeVector(&k);


  bool ans;
  DoubleWriter bw(binary ? "b,f,ark,scp:tmpf,tmpf.scp" : "t,f,ark,scp:tmpf,tmpf.scp");  // Putting the
  // "flush" option in too, just for good measure..
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);


  std::string name;
  if (sorted) name += "s,";
  else if (rand()%2 == 0) name += "ns,";
  if (called_sorted) name += "cs,";
  else if (rand()%2 == 0) name += "ncs,";
  if (once) name += "o,";
  else if (rand()%2 == 0) name += "no,";
  name += std::string(read_scp ? "scp:tmpf.scp" : "ark:tmpf");

  RandomAccessDoubleReader sbr(name);

  if (sz != 0) {
    std::vector<std::string> read_keys;
    int32 read_sz = rand() % 5;
    for (int32 i = 0; i < read_sz; i++)
      read_keys.push_back(k[rand() % k.size()]);
    std::sort(read_keys.begin(), read_keys.end());
    if (once) Uniq(&read_keys);
    if (!called_sorted)
      RandomizeVector(&read_keys);

    for (size_t i = 0; i < read_keys.size(); i++) {
      std::cout << "Looking up key " << read_keys[i] << std::endl;
      std::string cur_key = read_keys[i];
      double value;
      for (size_t i = 0; i < k.size(); i++)
        if (cur_key == k[i]) value = v[i];
      if (rand() % 2 == 0) {
        bool ans = sbr.HasKey(cur_key);
        KALDI_ASSERT(ans == true);
      }
      if (binary) {
        KALDI_ASSERT(value == sbr.Value(cur_key));
      } else {
        KALDI_ASSERT(ApproxEqual(value, sbr.Value(cur_key)));
      }
    }
  }
}


void UnitTestTableRandomBothDoubleMatrix(bool binary, bool read_scp,
                                         bool sorted, bool called_sorted,
                                         bool once) {
  int32 sz = rand() % 10;
  std::vector<std::string> k;
  std::vector<Matrix<double> > v;

  for (int32 i = 0; i < sz; i++) {
    k.push_back( CharToString( 'a' + static_cast<char>(i)));  // This gives us
    // some single quotes too but it doesn't really matter.
    if (i%2 == 0) k.back() = k.back() +  CharToString( 'a' + i);  // make them different lengths.
    v.resize(v.size()+1);
    v.back().Resize(1 + rand()%3, 1 + rand()%3);
    for (int32 j = 0; j < v.back().NumRows(); j++)
      for (int32 k = 0; k < v.back().NumCols(); k++)
        v.back()(j, k) =  (rand() % 100);
  }

  if (!sorted)
    RandomizeVector(&k);


  bool ans;
  DoubleMatrixWriter bw(binary ? "b,f,ark,scp:tmpf,tmpf.scp" : "t,f,ark,scp:tmpf,tmpf.scp");  // Putting the
  // "flush" option in too, just for good measure..
  for (int32 i = 0; i < sz; i++)  {
    bw.Write(k[i], v[i]);
  }
  ans = bw.Close();
  KALDI_ASSERT(ans);


  std::string name;
  if (sorted) name += "s,";
  else if (rand()%2 == 0) name += "ns,";
  if (called_sorted) name += "cs,";
  else if (rand()%2 == 0) name += "ncs,";
  if (once) name += "o,";
  else if (rand()%2 == 0) name += "no,";
  name += std::string(read_scp ? "scp:tmpf.scp" : "ark:tmpf");

  RandomAccessDoubleMatrixReader sbr(name);

  if (sz != 0) {
    std::vector<std::string> read_keys;
    int32 read_sz = rand() % 5;
    for (int32 i = 0; i < read_sz; i++)
      read_keys.push_back(k[rand() % k.size()]);
    std::sort(read_keys.begin(), read_keys.end());
    if (once) Uniq(&read_keys);
    if (!called_sorted)
      RandomizeVector(&read_keys);

    for (size_t i = 0; i < read_keys.size(); i++) {
      std::cout << "Looking up key " << read_keys[i] << std::endl;
      std::string cur_key = read_keys[i];
      Matrix<double> *value_ptr = NULL;
      for (size_t i = 0; i < k.size(); i++)
        if (cur_key == k[i]) value_ptr = &(v[i]);
      if (rand() % 2 == 0) {
        bool ans = sbr.HasKey(cur_key);
        KALDI_ASSERT(ans == true);
      }
      if (binary) {
        KALDI_ASSERT(value_ptr->ApproxEqual(sbr.Value(cur_key), 1.0e-10));
      } else {
        KALDI_ASSERT(value_ptr->ApproxEqual(sbr.Value(cur_key), 0.01));
      }
    }
  }
}



}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestReadScriptFile();
  UnitTestClassifyWspecifier();
  UnitTestClassifyRspecifier();
  for (int i = 0; i < 10; i++) {
    bool b = (i == 0);
    UnitTestTableSequentialBool(b);
    UnitTestTableSequentialInt32(b);
    UnitTestTableSequentialInt32Script(b);
    UnitTestTableSequentialDouble(b);
    for (int j = 0; j < 2; j++) {
      bool c = (j == 0);
      UnitTestTableSequentialDoubleBoth(b, c);
      UnitTestTableSequentialDoubleMatrixBoth(b, c);
      UnitTestTableSequentialInt32VectorBoth(b, c);
      UnitTestTableSequentialInt32PairVectorBoth(b, c);
      UnitTestTableSequentialInt32VectorVectorBoth(b, c);
      UnitTestTableSequentialBaseFloatVectorBoth(b, c);
      for (int k = 0; k < 2; k++) {
        bool d = (k == 0);
        for (int l = 0; l < 2; l++) {
          bool e = (l == 0);
          for (int m = 0; m < 2; m++) {
            bool f = (m == 0);
            UnitTestTableRandomBothDouble (b, c, d, e, f);
            UnitTestTableRandomBothDoubleMatrix (b, c, d, e, f);
          }
        }
      }
    }
  }
  std::cout << "Test OK.\n";
  return 0;
}

