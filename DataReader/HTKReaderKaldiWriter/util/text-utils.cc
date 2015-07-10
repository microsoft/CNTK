// util/text-utils.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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

#include "base/kaldi-common.h"
#include "util/text-utils.h"


namespace kaldi {


template<class F>
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings, // typically false
                         std::vector<F> *out) {
  KALDI_ASSERT(out != NULL);
  if ( *(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    F f = 0;
    if (!ConvertStringToReal(split[i], &f))
      return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<float> *out);
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<double> *out);

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

void JoinVectorToString(const std::vector<std::string> &vec_in,
                        const char *delim, bool omit_empty_strings,
                        std::string *str_out) {
  std::string tmp_str;
  for (size_t i = 0; i < vec_in.size(); i++) {
    if (!omit_empty_strings || !vec_in[i].empty()) {
      tmp_str.append(vec_in[i]);
      if (i < vec_in.size() - 1)
        if (!omit_empty_strings || !vec_in[i+1].empty())
          tmp_str.append(delim);
    }
  }
  str_out->swap(tmp_str);
}

void Trim(std::string *str) {
  const char *white_chars = " \t\n\r\f\v";

  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

bool IsToken(const std::string &token) {
  size_t l = token.length();
  if (l == 0) return false;
  for (size_t i = 0; i < l; i++) {
    char c = token[i];
    if ( (!isprint(c) || isspace(c)) && (isascii(c) || c == (char)255) ) return false;
    // The "&& (isascii(c) || c == 255)" was added so that we won't reject non-ASCII
    // characters such as French characters with accents [except for 255 which is
    // "nbsp", a form of space].
  }
  return true;
}


void SplitStringOnFirstSpace(const std::string &str,
                             std::string *first,
                             std::string *rest) {
  const char *white_chars = " \t\n\r\f\v";
  typedef std::string::size_type I;
  const I npos = std::string::npos;
  I first_nonwhite = str.find_first_not_of(white_chars);
  if (first_nonwhite == npos) {
    first->clear();
    rest->clear();
    return;
  }
  // next_white is first whitespace after first nonwhitespace.
  I next_white = str.find_first_of(white_chars, first_nonwhite);

  if (next_white == npos) {  // no more whitespace...
    *first = std::string(str, first_nonwhite);
    rest->clear();
    return;
  }
  I next_nonwhite = str.find_first_not_of(white_chars, next_white);
  if (next_nonwhite == npos) {
    *first = std::string(str, first_nonwhite, next_white-first_nonwhite);
    rest->clear();
    return;
  }

  I last_nonwhite = str.find_last_not_of(white_chars);
  KALDI_ASSERT(last_nonwhite != npos);  // or coding error.

  *first = std::string(str, first_nonwhite, next_white-first_nonwhite);
  *rest = std::string(str, next_nonwhite, last_nonwhite+1-next_nonwhite);
}

bool IsLine(const std::string &line) {
  if (line.find('\n') != std::string::npos) return false;
  if (line.empty()) return true;
  if (isspace(*(line.begin()))) return false;
  if (isspace(*(line.rbegin()))) return false;
  std::string::const_iterator iter = line.begin(), end = line.end();
  for (; iter!= end; iter++)
    if ( ! isprint(*iter)) return false;
  return true;
}

bool ConvertStringToReal(const std::string &str,
                         double *out) {
  const char *this_str = str.c_str();
  char *end = NULL;
  errno = 0;
  double d = KALDI_STRTOD(this_str, &end);
  if (end != this_str)
    while (isspace(*end)) end++;
  if (end == this_str || *end != '\0' || errno != 0)
    return false;
  *out = d;
  return true;
}

bool ConvertStringToReal(const std::string &str,
                         float *out) {
  const char *this_str = str.c_str();
  char *end = NULL;
  errno = 0;
  double d = KALDI_STRTOD(this_str, &end);
  if (end != this_str)
    while (isspace(*end)) end++;
  if (end == this_str || *end != '\0' || errno != 0)
    return false;
  *out = static_cast<float>(d);
  return true;
}

}  // end namespace kaldi
