// util/kaldi-table.h

// Copyright 2009-2011    Microsoft Corporation
//                2013    Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_UTIL_KALDI_TABLE_H_
#define KALDI_UTIL_KALDI_TABLE_H_

#include <string>
#include <vector>
#include <utility>

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"

namespace kaldi {

// Forward declarations
template<class Holder> class RandomAccessTableReaderImplBase;
template<class Holder>  class SequentialTableReaderImplBase;
template<class Holder>  class TableWriterImplBase;

/// \addtogroup table_group
/// @{

// This header defines the Table classes (RandomAccessTableReader,
// SequentialTableReader and TableWriter) and explains what the Holder classes,
// which the Table class requires as a template argument, are like.  It also
// explains the "rspecifier" and "wspecifier" concepts (these are strings that
// explain how to read/write objects via archives or scp files.  A table is
// conceptually a collection of objects of a particular type T indexed by keys
// of type std::string (these Keys additionally have an order within each table).
// The Table classes are templated on a type (call it Holder) such that Holder::T
// is a typedef equal to T.

// see kaldi-holder.h for detail on the Holder classes.

typedef std::vector<std::string> KeyList;

// Documentation for "wspecifier"
// "wspecifier" describes how we write a set of objects indexed by keys.
// The basic, unadorned wspecifiers are as follows:
//
//  ark:wxfilename
//  scp:rxfilename
//  ark,scp:filename,wxfilename
//  ark,scp:filename,wxfilename
//
//
//  We also allow the following modifiers:
//  t means text mode.
//  b means binary mode.
//  f means flush the stream after writing each entry.
//   (nf means don't flush, and isn't very useful as the default is to flush).
//  p means permissive mode, when writing to an "scp" file only: will ignore
//     missing scp entries, i.e. won't write anything for those files but will
//     return success status).
//
//  So the following are valid wspecifiers:
//  ark,b,f:foo
//  "ark,b,b:| gzip -c > foo"
//  "ark,scp,t,nf:foo.ark,|gzip -c > foo.scp.gz"
//  ark,b:-
//
//  The meanings of rxfilename and wxfilename are as described in
//  kaldi-stream.h (they are filenames but include pipes, stdin/stdout
//  and so on; filename is a regular filename.
//

//  The ark:wxfilename type of wspecifier instructs the class to
//  write directly to an archive.  For small objects (e.g. lists of ints),
//  the text archive format will generally be human readable with one line
//  per entry in the archive.
//
//  The type "scp:xfilename" refers to an scp file which should
//  already exist on disk, and tells us where to write the data for
//  each key (usually an actual file); each line of the scp file
//  would be:
//   key xfilename
//
//  The type ark,scp:filename,wxfilename means
//  we write both an archive and an scp file that specifies offsets into the
//  archive, with lines like:
//    key filename:12407
//  where the number is the byte offset into the file.
//  In this case we restrict the archive-filename to be an actual filename,
//  as we can't see a situtation where an extended filename would make sense
//  for this (we can't fseek() in pipes).

enum WspecifierType  {
  kNoWspecifier,
  kArchiveWspecifier,
  kScriptWspecifier,
  kBothWspecifier
};

struct WspecifierOptions {
  bool binary;
  bool flush;
  bool permissive; // will ignore absent scp entries.
  WspecifierOptions(): binary(true), flush(false), permissive(false) { }
};

// ClassifyWspecifier returns the type of the wspecifier string,
// and (if pointers are non-NULL) outputs the extra information
// about the options, and the script and archive
// filenames.
WspecifierType ClassifyWspecifier(const std::string &wspecifier,
                                  std::string *archive_wxfilename,
                                  std::string *script_wxfilename,
                                  WspecifierOptions *opts);

// ReadScriptFile reads an .scp file in its entirety, and appends it
// (in order as it was in the scp file) in script_out_, which contains
// pairs of (key, xfilename).  The .scp
// file format is: on each line, key xfilename
// where xfilename means rxfilename or wxfilename, and may contain internal spaces
// (we trim away any leading or trailing space).  The key is space-free.
// ReadScriptFile returns true if the format was valid (empty files
// are valid).
// If 'print_warnings', it will print out warning messages that explain what kind
// of error there was.
bool ReadScriptFile(const std::string &rxfilename,
                    bool print_warnings,
                    std::vector<std::pair<std::string, std::string> > *script_out);

// This version of ReadScriptFile works from an istream.
bool ReadScriptFile(std::istream &is,
                    bool print_warnings,
                    std::vector<std::pair<std::string, std::string> > *script_out);

// Writes, for each entry in script, the first element, then ' ', then the second
// element then '\n'.  Checks that the keys (first elements of pairs) are valid
// tokens (nonempty, no whitespace), and the values (second elements of pairs)
// are newline-free and contain no leading or trailing space.  Returns true on
// success.
bool WriteScriptFile(const std::string &wxfilename,
                     const std::vector<std::pair<std::string, std::string> > &script);

// This version writes to an ostream.
bool WriteScriptFile(std::ostream &os,
                     const std::vector<std::pair<std::string, std::string> > &script);

// Documentation for "rspecifier"
// "rspecifier" describes how we read a set of objects indexed by keys.
// The possibilities are:
//
// ark:rxfilename
// scp:rxfilename
//
// We also allow various modifiers:
//   o   means the program will only ask for each key once, which enables
//       the reader to discard already-asked-for values.
//   s   means the keys are sorted on input (means we don't have to read till
//       eof if someone asked for a key that wasn't there).
//   cs  means that it is called in sorted order (we are generally asserting this
//       based on knowledge of how the program works).
//   p   means "permissive", and causes it to skip over keys whose corresponding
//       scp-file entries cannot be read. [and to ignore errors in archives and
//       script files, and just consider the "good" entries].
//       We allow the negation of the options above, as in no, ns, np,
//       but these aren't currently very useful (just equivalent to omitting the
//       corresponding option).
//      [any of the above options can be prefixed by n to negate them, e.g. no, ns,
//       ncs, np; but these aren't currently useful as you could just omit the option].
//
//   b   is ignored [for scripting convenience]
//   t   is ignored [for scripting convenience]
//
//
//  So for instance the following would be a valid rspecifier:
//
//   "o, s, p, ark:gunzip -c foo.gz|"

struct  RspecifierOptions {
  // These options only make a difference for the RandomAccessTableReader class.
  bool once;   // we assert that the program will only ask for each key once.
  bool sorted;  // we assert that the keys are sorted.
  bool called_sorted;  // we assert that the (HasKey(), Value() functions will
  // also be called in sorted order.  [this implies "once" but not vice versa].
  bool permissive;  // If "permissive", when reading from scp files it treats
  // scp files that can't be read as if the corresponding key were not there.
  // For archive files it will suppress errors getting thrown if the archive
  
  // is corrupted and can't be read to the end.

  RspecifierOptions(): once(false), sorted(false),
                       called_sorted(false), permissive(false) { }
};

enum RspecifierType  {
  kNoRspecifier,
  kArchiveRspecifier,
  kScriptRspecifier
};

RspecifierType ClassifyRspecifier(const std::string &rspecifier, std::string *rxfilename,
                                  RspecifierOptions *opts);

// Class Table<Holder> is useful when you want the entire set of
// objects in memory.  NOT IMPLEMENTED YET.
// It is the least scalable way of accessing data in Tables.
// The *TableReader and TableWriter classes are more scalable.


/// Allows random access to a collection
/// of objects in an archive or script file; see \ref io_sec_tables.
template<class Holder>
class RandomAccessTableReader {
 public:
  typedef typename Holder::T T;

  RandomAccessTableReader(): impl_(NULL) { }

  // This constructor equivalent to default constructor + "open", but
  // throws on error.
  RandomAccessTableReader(const std::string &rspecifier);

  // Opens the table.
  bool Open(const std::string &rspecifier);

  // Returns true if table is open.
  bool IsOpen() const { return (impl_ != NULL); }

  // Close() will close the table [throws if it was not open],
  // and returns true on success (false if we were reading an
  // archive and we discovered an error in the archive).
  bool Close();

  // Says if it has this key.
  // If you are using the "permissive" (p) read option,
  // it will return false for keys whose corresponding entry
  // in the scp file cannot be read.

  bool HasKey(const std::string &key);

  // Value() may throw if you are reading an scp file, you
  // do not have the "permissive" (p) option, and an entry
  // in the scp file cannot be read.  Typically you won't
  // want to catch this error.
  const T &Value(const std::string &key);

  ~RandomAccessTableReader();

 private:
  void CheckImpl() const; // Checks that impl_ is non-NULL; prints an error
                          // message and dies (with KALDI_ERR) if NULL.
  RandomAccessTableReaderImplBase<Holder> *impl_;
};



/// A templated class for reading objects sequentially from an archive or script
/// file; see \ref io_sec_tables.
template<class Holder>
class SequentialTableReader {
 public:
  typedef typename Holder::T T;

  SequentialTableReader(): impl_(NULL) { }

  // This constructor equivalent to default constructor + "open", but
  // throws on error.
  SequentialTableReader(const std::string &rspecifier);

  // Opens the table.  Returns exit status; but does throw if previously
  // open stream was in error state.  Call Close to stop this [anyway,
  // calling Open more than once is not recommended.]
  bool Open(const std::string &rspecifier);

  // Returns true if we're done.  It will also return true if there's some kind
  // of error and we can't read any more; in this case, you can detect the
  // error by calling Close and checking the return status; otherwise
  // the destructor will throw.
  inline bool Done();

  // Only valid to call Key() if Done() returned false.
  inline std::string Key();

  // FreeCurrent() is provided as an optimization to save memory, for large
  // objects.  It instructs the class to deallocate the current value. The
  // reference Value() will/ be invalidated by this.

  void FreeCurrent();

  // Return reference to the current value.
  // The reference is valid till next call to this object.
  // If will throw if you are reading an scp file, did not
  // specify the "permissive" (p) option and the file cannot
  // be read.  [The permissive option makes it behave as if that
  // key does not even exist, if the corresponding file cannot be
  // read.]  You probably wouldn't want to catch this exception;
  // the user can just specify the p option in the rspecifier.
  const T &Value();

  // Next goes to the next key.  It will not throw; any error will
  // result in Done() returning true, and then the destructor will
  // throw unless you call Close().
  void Next();

  // Returns true if table is open for reading (does not imply
  // stream is in good state).
  bool IsOpen() const;

  // Close() will return false (failure) if Done() became true
  // because of an error/ condition rather than because we are
  // really done [e.g. because of an error or early termination
  // in the archive].
  // If there is an error and you don't call Close(), the destructor
  // will fail.
  // Close()
  bool Close();

  // The destructor may throw.  This is the desired behaviour, as it's the way we
  // signal the error to the user (to detect it, call Close().  The issue is that
  // otherwise the user has no way to tell whether Done() returned true because
  // we reached the end of the archive or script, or because there was an error
  // that prevented further reading.
  ~SequentialTableReader();
 private:
  void CheckImpl() const; // Checks that impl_ is non-NULL; prints an error
                          // message and dies (with KALDI_ERR) if NULL.
  SequentialTableReaderImplBase<Holder> *impl_;
};


/// A templated class for writing objects to an
/// archive or script file; see \ref io_sec_tables.
template<class Holder>
class TableWriter {
 public:
  typedef typename Holder::T T;

  TableWriter(): impl_(NULL) { }

  // This constructor equivalent to default constructor
  // + "open", but throws on error.  See docs for
  // wspecifier above.
  TableWriter(const std::string &wspecifier);

  // Opens the table.  See docs for wspecifier above.
  // If it returns true, it is open.
  bool Open(const std::string &wspecifier);

  // Returns true if open for writing.
  bool IsOpen() const;

  // Write the object.  Throws  std::runtime_error on error (via the
  // KALDI_ERR macro)
  inline void Write(const std::string &key, const T &value) const;


  // Flush will flush any archive; it does not return error status
  // or throw, any errors will be reported on the next Write or Close.
  // Useful if we may be writing to a command in a pipe and want
  // to ensure good CPU utilization.
  void Flush();

  // Close() is not necessary to call, as the destructor
  // closes it; it's mainly useful if you want to handle
  // error states because the destructor will throw on
  // error if you do not call Close().
  bool Close();

  ~TableWriter();
 private:
  void CheckImpl() const; // Checks that impl_ is non-NULL; prints an error
                          // message and dies (with KALDI_ERR) if NULL.
  TableWriterImplBase<Holder> *impl_;
};


/// This class is for when you are reading something in random access, but
/// it may actually be stored per-speaker (or something similar) but the 
/// keys you're using are per utterance.  So you also provide an "rxfilename"
/// for a file containing lines like
/// utt1 spk1
/// utt2 spk1
/// utt3 spk1
/// and so on.  Note: this is optional; if it is an empty string, we just won't
/// do the mapping.  Also, "table_rxfilename" may be the empty string (as for
/// a regular table), in which case the table just won't be opened.
/// We provide only the most frequently used of the functions of RandomAccessTableReader.

template<class Holder>
class RandomAccessTableReaderMapped {
 public:
  typedef typename Holder::T T;
  /// Note: "utt2spk_rxfilename" will in the normal case be an rxfilename
  /// for an utterance to speaker map, but this code is general; it accepts
  /// a generic map.
  RandomAccessTableReaderMapped(const std::string &table_rxfilename,
                                const std::string &utt2spk_rxfilename);

  RandomAccessTableReaderMapped() {};

  /// Note: when calling Open, utt2spk_rxfilename may be empty.
  bool Open(const std::string &table_rxfilename,
            const std::string &utt2spk_rxfilename);

  bool HasKey(const std::string &key);
  const T &Value(const std::string &key);
  inline bool IsOpen() const { return reader_.IsOpen(); }
  inline bool Close() { return reader_.Close(); }
  
  // Use the default destructor.
 private:
  RandomAccessTableReader<Holder> reader_;
  RandomAccessTableReader<TokenHolder> token_reader_;
  std::string utt2spk_rxfilename_; // Used only in diagnostic messages.
};


/// @} end "addtogroup table_group"
} // end namespace kaldi

#include "kaldi-table-inl.h"

#endif  // KALDI_UTIL_KALDI_TABLE_H_
