// util/kaldi-table-inl.h

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


#ifndef KALDI_UTIL_KALDI_TABLE_INL_H_
#define KALDI_UTIL_KALDI_TABLE_INL_H_

#include <algorithm>
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "util/stl-utils.h" // for StringHasher.


namespace kaldi {

/// \addtogroup table_impl_types
/// @{

template<class Holder> class SequentialTableReaderImplBase {
 public:
  typedef typename Holder::T T;
  // note that Open takes rxfilename not rspecifier.
  virtual bool Open(const std::string &rxfilename) = 0;
  virtual bool Done() const = 0;
  virtual bool IsOpen() const = 0;
  virtual std::string Key() = 0;
  virtual const T &Value() = 0;
  virtual void FreeCurrent() = 0;
  virtual void Next() = 0;
  virtual bool Close() = 0;
  virtual ~SequentialTableReaderImplBase() { }
};


// This is the implementation for SequentialTableReader
// when it's actually a script file.
template<class Holder>  class SequentialTableReaderScriptImpl:
      public SequentialTableReaderImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  SequentialTableReaderScriptImpl(): state_(kUninitialized) { }

  virtual bool Open(const std::string &rspecifier) {
    if (state_ != kUninitialized)
      if (! Close()) // call Close() yourself to suppress this exception.
        KALDI_ERR << "TableReader::Open, error closing previous input "
                  << PrintableRxfilename(script_rxfilename_);
    bool binary;
    RspecifierType rs = ClassifyRspecifier(rspecifier, &script_rxfilename_,
                                           &opts_);
    KALDI_ASSERT(rs == kScriptRspecifier);
    if (!script_input_.Open(script_rxfilename_, &binary)) {  // Failure on Open
      KALDI_WARN << "Failed to open script file "
                 << PrintableRxfilename(script_rxfilename_);
      state_ = kUninitialized;
      return false;
    } else {  // Open succeeded.
      if (binary) {  // script file should not be binary file..
        state_ = kError;  // bad script file.
        script_input_.Close();
        return false;
      } else {
        state_ = kFileStart;
        Next();
        if (state_ == kError) {
          script_input_.Close();
          return false;
        }
        if (opts_.permissive) {  // Next() will have preloaded.
          KALDI_ASSERT(state_ == kLoadSucceeded || state_ == kEof);
        } else {
          KALDI_ASSERT(state_ == kHaveScpLine || state_ == kEof);
        }
        return true;  // Success.
      }
    }
  }

  virtual bool IsOpen() const {
    switch (state_) {
      case kEof: case kError: case kHaveScpLine: case kLoadSucceeded: case kLoadFailed: return true;
      case kUninitialized:  return false;
      default: KALDI_ERR << "IsOpen() called on invalid object.";  // kFileStart is not valid
        // state for user to call something on.
        return false;
    }
  }

  virtual bool Done() const {
    switch (state_) {
      case kHaveScpLine: return false;
      case kLoadSucceeded: case kLoadFailed: return false;
        // These cases are because we want LoadCurrent()
        // to be callable after Next() and to not change the Done() status [only Next() should change
        // the Done() status].
      case kEof: case kError: return true;  // Error condition, like Eof, counts as Done(); the destructor
        // or Close() will inform the user of the error.
      default: KALDI_ERR << "Done() called on TableReader object at the wrong time.";
        return false;
    }
  }

  virtual std::string Key() {
    // Valid to call this whenever Done() returns false.
    switch (state_) {
      case kHaveScpLine: case kLoadSucceeded: case kLoadFailed: break;
      default:
        // coding error.
        KALDI_ERR << "Key() called on TableReader object at the wrong time.";
    }
    return key_;
  }
  const T &Value() {
    StateType orig_state = state_;
    if (state_ == kHaveScpLine) LoadCurrent();  // Takes
    // state_ to kLoadSucceeded or kLoadFailed.
    if (state_ == kLoadFailed) {  // this can happen due to
      // a file listed in an scp file not existing, or
      // read failure, failure of a command, etc.
      if (orig_state == kHaveScpLine)
        KALDI_ERR << "TableReader: failed to load object from "
                  << PrintableRxfilename(data_rxfilename_)
                  << " (to suppress this error, add the permissive "
                  << "(p, ) option to the rspecifier.";

      else // orig_state_ was kLoadFailed, which only could have happened
        // if the user called FreeCurrent().
        KALDI_ERR << "TableReader: you called Value() after FreeCurrent().";
    } else if (state_ != kLoadSucceeded) {
      // This would be a coding error.
      KALDI_ERR << "TableReader: Value() called at the wrong time.";
    }
    return holder_.Value();
  }
  void FreeCurrent() {
    if (state_ == kLoadSucceeded) {
      holder_.Clear();
      state_ = kLoadFailed;
    } else {
      KALDI_WARN << "TableReader: FreeCurrent called at the wrong time.";
    }
  }
  void Next() {
    while (1) {
      NextScpLine();
      if (Done()) return;
      if (opts_.permissive) {
        // Permissive mode means, when reading scp files, we treat keys whose scp entry
        // cannot be read as nonexistent.  This means trying to read.
        if (LoadCurrent()) return;  // Success.
        // else try the next scp line.
      } else {
        return;  // We go the next key; Value() will crash if we can't
        // read the scp line.
      }
    }
  }

  virtual bool Close() {
    // Close() will succeed if the stream was not in an error
    // state.  To clean up, it also closes the Input objects if
    // they're open.
    if (script_input_.IsOpen())
      script_input_.Close();
    if (data_input_.IsOpen())
      data_input_.Close();
    if (state_ == kLoadSucceeded)
      holder_.Clear();
    if (!this->IsOpen())
      KALDI_ERR << "Close() called on input that was not open.";
    StateType old_state = state_;
    state_ = kUninitialized;
    if (old_state == kError) {
      if (opts_.permissive) {
        KALDI_WARN << "Close() called on scp file with read error, ignoring the "
            "error because permissive mode specified.";
        return true;
      } else  return false;  // User will do something with the error status.
    } else  return true;
  }

  virtual ~SequentialTableReaderScriptImpl() {
    if (state_ == kError)
      KALDI_ERR << "TableReader: reading script file failed: from scp "
                << PrintableRxfilename(script_rxfilename_);
    // If you don't want this exception to be thrown you can
    // call Close() and check the status.
    if (state_ == kLoadSucceeded)
      holder_.Clear();
  }
 private:
  bool LoadCurrent() {
    // Attempts to load object whose rxfilename is on the current scp line.
    if (state_ != kHaveScpLine)
      KALDI_ERR << "TableReader: LoadCurrent() called at the wrong time.";
    bool ans;
    // note, NULL means it doesn't read the binary-mode header
    if (Holder::IsReadInBinary()) ans = data_input_.Open(data_rxfilename_, NULL);
    else ans = data_input_.OpenTextMode(data_rxfilename_);
    if (!ans) {
      // May want to make this warning a VLOG at some point
      KALDI_WARN << "TableReader: failed to open file "
                 << PrintableRxfilename(data_rxfilename_);
      state_ = kLoadFailed;
      return false;
    } else {
      if (holder_.Read(data_input_.Stream())) {
        state_ = kLoadSucceeded;
        return true;
      } else {  // holder_ will not contain data.
        KALDI_WARN << "TableReader: failed to load object from "
                   << PrintableRxfilename(data_rxfilename_);
        state_ = kLoadFailed;
        return false;
      }
    }
  }

  // Reads the next line in the script file.
  void NextScpLine() {
    switch (state_) {
      case kLoadSucceeded: holder_.Clear(); break;
      case kHaveScpLine: case kLoadFailed: case kFileStart: break;
      default:
        // No other states are valid to call Next() from.
        KALDI_ERR << "SequentialTableReader, reading script file: Next called wrongly.";
    }
    std::string line;
    if (getline(script_input_.Stream(), line)) {
      SplitStringOnFirstSpace(line, &key_, &data_rxfilename_);
      if (!key_.empty() && !data_rxfilename_.empty()) {
        // Got a valid line.
        state_ = kHaveScpLine;
      } else {
        // Got an invalid line.
        state_ = kError;  // we can't make sense of this
        // scp file and will now die.
      }
    } else {
      state_ = kEof;  // nothing more in the scp file.
      // Might as well close the input streams as don't need them.
      script_input_.Close();
      if (data_input_.IsOpen())
        data_input_.Close();
    }
  }


  Input script_input_;  // Input object for the .scp file
  Input data_input_;   // Input object for the entries in
  // the script file.
  Holder holder_;  // Holds the object.
  bool binary_;  // Binary-mode archive.
  std::string key_;
  std::string script_rxfilename_;  // of the script file.
  RspecifierOptions opts_;  // options.
  std::string data_rxfilename_;  // of the file we're reading.
  enum StateType {
    //       [The state of the reading process]               [does holder_ [is script_inp_
    //                                                         have object]   open]
    kUninitialized,  // Uninitialized or closed.                    no         no
    kEof,     // We did Next() and found eof in script file.       no         no
    kError,   // Some other error                                  no         yes
    kHaveScpLine,  // Just called Open() or Next() and have a       no         yes
    // line of the script file but no data.
    kLoadSucceeded,  // Called LoadCurrent() and it succeeded.     yes         yes
    kLoadFailed,  // Called LoadCurrent() and it failed,           no         yes
    // or the user called FreeCurrent().. note,
    // if when called by user we are in this state,
    // it means the user called FreeCurrent().
    kFileStart,        // [state we only use internally]           no         yes
  } state_;
};


// This is the implementation for SequentialTableReader
// when it's an archive.  Note that the archive format is:
// key1 [space] object1 key2 [space]
// object2 ... eof.
// "object1" is the output of the Holder::Write function and will
// typically contain a binary header (in binary mode) and then
// the output of object.Write(os, binary).
// The archive itself does not care whether it is in binary
// or text mode, for reading purposes.

template<class Holder>  class SequentialTableReaderArchiveImpl:
      public SequentialTableReaderImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  SequentialTableReaderArchiveImpl(): state_(kUninitialized) { }

  virtual bool Open(const std::string &rspecifier) {
    if (state_ != kUninitialized) {
      if (! Close()) {  // call Close() yourself to suppress this exception.
        if (opts_.permissive)
          KALDI_WARN << "TableReader::Open, error closing previous input "
              "(only warning, since permissive mode).";
        else
          KALDI_ERR << "TableReader::Open, error closing previous input.";
      }
    }
    RspecifierType rs = ClassifyRspecifier(rspecifier, &archive_rxfilename_,
                                           &opts_);
    KALDI_ASSERT(rs == kArchiveRspecifier);

    bool ans;
    // NULL means don't expect binary-mode header
    if (Holder::IsReadInBinary())
      ans = input_.Open(archive_rxfilename_, NULL);
    else
      ans = input_.OpenTextMode(archive_rxfilename_);
    if (!ans) {  // header.
      KALDI_WARN << "TableReader: failed to open stream "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kUninitialized;  // Failure on Open
      return false;  // User should print the error message.
    }
    state_ = kFileStart;
    Next();
    if (state_ == kError) {
      KALDI_WARN << "TableReader: error beginning to read table (wrong filename?): "
                 << PrintableRxfilename(archive_rxfilename_);
      input_.Close();
      state_ = kUninitialized;
      return false;
    }
    KALDI_ASSERT(state_ == kHaveObject || state_ == kEof);
    return true;
  }

  virtual void Next() {
    switch (state_) {
      case kHaveObject:
        holder_.Clear(); break;
      case kFileStart: case kFreedObject:
        break;
      default:
        KALDI_ERR << "TableReader: Next() called wrongly.";
    }
    std::istream &is = input_.Stream();
    is.clear();  // Clear any fail bits that may have been set... just in case
    // this happened in the Read function.
    is >> key_;  // This eats up any leading whitespace and gets the string.
    if (is.eof()) {
      state_ = kEof;
      return;
    }
    if (is.fail()) {  // This shouldn't really happen, barring file-system errors.
      KALDI_WARN << "Error reading archive "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      return;
    }
    int c;
    if ((c = is.peek()) != ' ' && c != '\t' && c != '\n') {  // We expect a space ' ' after the key.
      // We also allow tab [which is consumed] and newline [which is not], just
      // so we can read archives generated by scripts that may not be fully
      // aware of how this format works.
      KALDI_WARN << "Invalid archive file format: expected space after key "
                 << key_ << ", got character "
                 << CharToString(static_cast<char>(is.peek())) << ", reading "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      return;
    }
    if (c != '\n') is.get();  // Consume the space or tab.
    if (holder_.Read(is)) {
      state_ = kHaveObject;
      return;
    } else {
      KALDI_WARN << "Object read failed, reading archive "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      return;
    }
  }

  virtual bool IsOpen() const {
    switch (state_) {
      case kEof: case kError: case kHaveObject: case kFreedObject: return true;
      case kUninitialized: return false;
      default: KALDI_ERR << "IsOpen() called on invalid object.";  // kFileStart is not valid
        // state for user to call something on.
        return false;
    }
  }

  virtual bool Done() const {
    switch (state_) {
      case kHaveObject:
        return false;
      case kEof: case kError:
        return true;  // Error-state counts as Done(), but destructor
        // will fail (unless you check the status with Close()).
      default:
        KALDI_ERR << "Done() called on TableReader object at the wrong time.";
        return false;
    }
  }

  virtual std::string Key() {
    // Valid to call this whenever Done() returns false
    switch (state_) {
      case kHaveObject: break;  // only valid case.
      default:
        // coding error.
        KALDI_ERR << "Key() called on TableReader object at the wrong time.";
    }
    return key_;
  }
  const T &Value() {
    switch (state_) {
      case kHaveObject:
        break;  // only valid case.
      default:
        // coding error.
        KALDI_ERR << "Value() called on TableReader object at the wrong time.";
    }
    return holder_.Value();
  }
  virtual void FreeCurrent() {
    if (state_ == kHaveObject) {
      holder_.Clear();
      state_ = kFreedObject;
    } else
      KALDI_WARN << "TableReader: FreeCurernt called at the wrong time.";
  }

  virtual bool Close() {
    if (! this->IsOpen())
      KALDI_ERR << "Close() called on TableReader twice or otherwise wrongly.";
    if (input_.IsOpen())
      input_.Close();
    if (state_ == kHaveObject)
      holder_.Clear();
    bool ans;
    if (opts_.permissive) {
      ans = true;  // always return success.
      if (state_ == kError)
        KALDI_WARN << "Error detected closing TableReader for archive "
                   << PrintableRxfilename(archive_rxfilename_) << " but ignoring "
                   << "it as permissive mode specified.";
    } else
      ans = (state_ != kError);  // If error state, user should detect it.
    state_ = kUninitialized;
    return ans;
  }

  virtual ~SequentialTableReaderArchiveImpl() {
    if (state_ == kError) {
      if (opts_.permissive)
        KALDI_WARN << "Error detected closing TableReader for archive "
                   << PrintableRxfilename(archive_rxfilename_) << " but ignoring "
                   << "it as permissive mode specified.";
      else
        KALDI_ERR << "TableReader: reading archive failed: "
                  << PrintableRxfilename(archive_rxfilename_);
    }
    // If you don't want this exception to be thrown you can
    // call Close() and check the status.
    if (state_ == kHaveObject)
      holder_.Clear();
  }
 private:
  Input input_;  // Input object for the archive
  Holder holder_;     // Holds the object.
  std::string key_;
  std::string archive_rxfilename_;
  RspecifierOptions opts_;
  enum {  //  [The state of the reading process]               [does holder_ [is input_
    //                                                         have object]   open]
    kUninitialized,  // Uninitialized or closed.                    no         no
    kFileStart,      // [state we use internally: just opened.]    no         yes
    kEof,     // We did Next() and found eof in archive            no         no
    kError,   // Some other error                                  no         no
    kHaveObject,  // We read the key and the object after it.       yes        yes
    kFreedObject,  // The user called FreeCurrent().                no         yes
  } state_;
};


template<class Holder>
SequentialTableReader<Holder>::SequentialTableReader(const std::string &rspecifier): impl_(NULL) {
  if (rspecifier != "" && !Open(rspecifier))
    KALDI_ERR << "Error constructing TableReader: rspecifier is " << rspecifier;
}

template<class Holder>
bool SequentialTableReader<Holder>::Open(const std::string &rspecifier) {
  if (IsOpen())
    if (!Close())
      KALDI_ERR << "SequentialTableReader<Holder>::Open(), could not close previously open object.";
  // now impl_ will be NULL.

  RspecifierType wt = ClassifyRspecifier(rspecifier, NULL, NULL);
  switch (wt) {
    case kArchiveRspecifier:
      impl_ = new SequentialTableReaderArchiveImpl<Holder>();
      break;
    case kScriptRspecifier:
      impl_ = new SequentialTableReaderScriptImpl<Holder>();
      break;
    case kNoRspecifier: default:
      KALDI_WARN << "Invalid rspecifier " << rspecifier;
      return false;
  }
  if (!impl_->Open(rspecifier)) {
    delete impl_;
    impl_ = NULL;
    return false;  // sub-object will have printed warnings.
  }
  else return true;
}

template<class Holder>
bool SequentialTableReader<Holder>::Close() {
  CheckImpl();  
  bool ans = impl_->Close();
  delete impl_;  // We don't keep around empty impl_ objects.
  impl_ = NULL;
  return ans;
}


template<class Holder>
bool SequentialTableReader<Holder>::IsOpen() const {
  return (impl_ != NULL);  // Because we delete the object whenever
  // that object is not open.  Thus, the IsOpen functions of the
  // Impl objects are not really needed.
}

template<class Holder>
std::string SequentialTableReader<Holder>::Key() {
  CheckImpl();
  return impl_->Key();  // this call may throw if called wrongly in other ways,
  // e.g. eof.
}


template<class Holder>
void SequentialTableReader<Holder>::FreeCurrent() {
  CheckImpl();
  impl_->FreeCurrent();
}


template<class Holder>
const typename SequentialTableReader<Holder>::T &
SequentialTableReader<Holder>::Value() {
  CheckImpl();
  return impl_->Value();  // This may throw (if LoadCurrent() returned false you are safe.).
}


template<class Holder>
void SequentialTableReader<Holder>::Next() {
  CheckImpl();
  impl_->Next();
}

template<class Holder>
bool SequentialTableReader<Holder>::Done() {
  CheckImpl();
  return impl_->Done();
}


template<class Holder>
SequentialTableReader<Holder>::~SequentialTableReader() {
  if (impl_)  delete impl_;
  // Destructor of impl_ may throw.
}



template<class Holder> class TableWriterImplBase {
 public:
  typedef typename Holder::T T;

  virtual bool Open(const std::string &wspecifier) = 0;

  // Write returns true on success, false on failure, but
  // some errors may not be detected until we call Close().
  // It throws (via KALDI_ERR) if called wrongly.  We could
  // have just thrown on all errors, since this is what
  // TableWriter does; it was designed this way because originally
  // TableWriter::Write returned an exit status.
  virtual bool Write(const std::string &key, const T &value) = 0;

  // Flush will flush any archive; it does not return error status,
  //  any errors will be reported on the next Write or Close.
  virtual void Flush() = 0;

  virtual bool Close() = 0;

  virtual bool IsOpen() const = 0;

  // May throw on write error if Close was not called.
  virtual ~TableWriterImplBase() { }
};


// The implementation of TableWriter we use when writing directly
// to an archive with no associated scp.
template<class Holder>
class TableWriterArchiveImpl: public TableWriterImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  virtual bool Open(const std::string &wspecifier) {
    switch (state_) {
      case kUninitialized:
        break;
      case kWriteError:
        KALDI_ERR << "TableWriter: opening stream, already open with write error.";
      case kOpen: default:
        if (!Close())  // throw because this error may not have been previously
          // detected by the user.
          KALDI_ERR << "TableWriter: opening stream, error closing previously open stream.";
    }
    WspecifierType ws = ClassifyWspecifier(wspecifier,
                                           &archive_wxfilename_,
                                           NULL,
                                           &opts_);
    KALDI_ASSERT(ws == kArchiveWspecifier);  // or wrongly called.

    if (output_.Open(archive_wxfilename_, opts_.binary, false)) {  // false means no binary header.
      state_ = kOpen;
      return true;
    } else {
      // stream will not be open.  User will report this error
      // (we return bool), so don't bother printing anything.
      state_ = kUninitialized;
      return false;
    }
  }

  virtual bool IsOpen() const {
    switch (state_) {
      case kUninitialized: return false;
      case kOpen: case kWriteError: return true;
      default: KALDI_ERR << "IsOpen() called on TableWriter in invalid state.";
    }
    return false;
  }

  // Write returns true on success, false on failure, but
  // some errors may not be detected till we call Close().
  virtual bool Write(const std::string &key, const T &value) {
    switch (state_) {
      case kOpen: break;
      case kWriteError:
        // user should have known from the last
        // call to Write that there was a problem.
        KALDI_WARN << "TableWriter: attempting to write to invalid stream.";
        return false;
      case kUninitialized: default:
        KALDI_ERR << "TableWriter: Write called on invalid stream";

    }
    // state is now kOpen or kWriteError.
    if (!IsToken(key)) // e.g. empty string or has spaces...
      KALDI_ERR << "TableWriter: using invalid key " << key;
    output_.Stream() << key << ' ';
    if (!Holder::Write(output_.Stream(), opts_.binary, value)) {
      KALDI_WARN << "TableWriter: write failure to "
                 << PrintableWxfilename(archive_wxfilename_);
      state_ = kWriteError;
      return false;
    }
    if (state_ == kWriteError) return false;  // Even if this Write seems to have
    // succeeded, we fail because a previous Write failed and the archive may be
    // corrupted and unreadable.

    if (opts_.flush)
      Flush();
    return true;
  }

  // Flush will flush any archive; it does not return error status,
  //  any errors will be reported on the next Write or Close.
  virtual void Flush() {
    switch (state_) {
      case kWriteError: case kOpen:
        output_.Stream().flush();  // Don't check error status.
        return;
      default:
        KALDI_WARN << "TableWriter: Flush called on not-open writer.";
    }
  }

  virtual bool Close() {
    if (!this->IsOpen() || !output_.IsOpen())
      KALDI_ERR << "TableWriter: Close called on a stream that was not open." << this->IsOpen() << ", " << output_.IsOpen();
    bool close_success = output_.Close();
    if (!close_success) {
      KALDI_WARN << "TableWriter: error closing stream: "
                 << PrintableWxfilename(archive_wxfilename_);
      state_ = kUninitialized;
      return false;
    }
    if (state_ == kWriteError) {
      KALDI_WARN << "TableWriter: closing writer in error state: "
                 << PrintableWxfilename(archive_wxfilename_);
      state_ = kUninitialized;
      return false;
    }
    state_ = kUninitialized;
    return true;
  }

  TableWriterArchiveImpl(): state_(kUninitialized) {}

  // May throw on write error if Close was not called.
  virtual ~TableWriterArchiveImpl() {
    if (!IsOpen()) return;
    else if (!Close())
      KALDI_ERR << "At TableWriter destructor: Write failed or stream close failed: "
                << PrintableWxfilename(archive_wxfilename_);
  }

 private:
  Output output_;
  WspecifierOptions opts_;
  std::string archive_wxfilename_;
  enum {               // is stream open?
    kUninitialized,    // no
    kOpen,             // yes
    kWriteError,       // yes
  } state_;
};




// The implementation of TableWriter we use when writing to
// individual files (more generally, wxfilenames) specified
// in an scp file that we read.

// Note: the code for this class is similar to RandomAccessTableReaderScriptImpl;
// try to keep them in sync.

template<class Holder>
class TableWriterScriptImpl: public TableWriterImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  TableWriterScriptImpl(): last_found_(0), state_(kUninitialized) {}

  virtual bool Open(const std::string &wspecifier) {
    switch (state_) {
      case kReadScript:
        KALDI_ERR << " Opening already open TableWriter: call Close first.";
      case kUninitialized: case kNotReadScript:
        break;
    }
    WspecifierType ws = ClassifyWspecifier(wspecifier,
                                           NULL,
                                           &script_rxfilename_,
                                           &opts_);
    KALDI_ASSERT(ws == kScriptWspecifier);  // or wrongly called.
    KALDI_ASSERT(script_.empty());  // no way it could be nonempty at this point.

    if (! ReadScriptFile(script_rxfilename_,
                         true,  // print any warnings
                         &script_)) {  // error reading script file or invalid format
      state_ = kNotReadScript;
      return false;  // no need to print further warnings.  user gets the error.
    }
    std::sort(script_.begin(), script_.end());
    for (size_t i = 0; i+1 < script_.size(); i++) {
      if (script_[i].first.compare(script_[i+1].first) >= 0) {
        // script[i] not < script[i+1] in lexical order...
        KALDI_WARN << "Script file " << PrintableRxfilename(script_rxfilename_)
                   << " contains duplicate key " << script_[i].first;
        state_ = kNotReadScript;
        return false;
      }
    }
    state_ = kReadScript;
    return true;
  }

  virtual bool IsOpen() const {  return (state_ == kReadScript);  }

  virtual bool Close() {
    if (!IsOpen())
      KALDI_ERR << "Close() called on TableWriter that was not open.";
    state_ = kUninitialized;
    last_found_ = 0;
    script_.clear();
    return true;
  }

  // Write returns true on success, false on failure, but
  // some errors may not be detected till we call Close().
  virtual bool Write(const std::string &key, const T &value) {
    if (!IsOpen())
      KALDI_ERR << "TableWriter: Write called on invalid stream";

    if (!IsToken(key)) // e.g. empty string or has spaces...
      KALDI_ERR << "TableWriter: using invalid key " << key;

    std::string wxfilename;
    if (!LookupFilename(key, &wxfilename)) {
      if (opts_.permissive) {
        return true; // In permissive mode, it's as if we're writing to /dev/null
                     // for missing keys.
      } else {
        KALDI_WARN << "TableWriter: script file "
                   << PrintableRxfilename(script_rxfilename_)
                   << " has no entry for key "<<key;
        return false;
      }
    }
    Output output;
    if (!output.Open(wxfilename, opts_.binary, false)) {
      // Open in the text/binary mode (on Windows) given by member var. "binary"
      // (obtained from wspecifier), but do not put the binary-mode header (it
      // will be written, if needed, by the Holder::Write function.)
      KALDI_WARN << "TableWriter: failed to open stream: "
                 << PrintableWxfilename(wxfilename);
      return false;
    }
    if (!Holder::Write(output.Stream(), opts_.binary, value)
        || !output.Close()) {
      KALDI_WARN << "TableWriter: failed to write data to "
                 << PrintableWxfilename(wxfilename);
      return false;
    }
    return true;
  }

  // Flush does nothing in this implementation, there is nothing to flush.
  virtual void Flush() { }


  virtual ~TableWriterScriptImpl() {
    // Nothing to do in destructor.
  }

 private:
  // Note: this function is almost the same as in RandomAccessTableReaderScriptImpl.
  bool LookupFilename(const std::string &key, std::string *wxfilename) {
    // First, an optimization: if we're going consecutively, this will
    // make the lookup very fast.
    last_found_++;
    if (last_found_ < script_.size() && script_[last_found_].first == key) {
      *wxfilename = script_[last_found_].second;
      return true;
    }
    std::pair<std::string, std::string> pr(key, "");  // Important that ""
    // compares less than or equal to any string, so lower_bound points to the
    // element that has the same key.
    typedef typename std::vector<std::pair<std::string, std::string> >::const_iterator 
        IterType;
    IterType iter = std::lower_bound(script_.begin(), script_.end(), pr);
    if (iter != script_.end() && iter->first == key) {
      last_found_ = iter - script_.begin();
      *wxfilename = iter->second;
      return true;
    } else {
      return false;
    }
  }


  WspecifierOptions opts_;
  std::string script_rxfilename_;

  // the script_ variable contains pairs of (key, filename), sorted using
  // std::sort.  This can be used with binary_search to look up filenames for
  // writing.  If this becomes inefficient we can use std::unordered_map (but I
  // suspect this wouldn't be significantly faster & would use more memory).
  // If memory becomes a problem here, the user should probably be passing
  // only the relevant part of the scp file rather than expecting us to get too
  // clever in the code.
  std::vector<std::pair<std::string, std::string> > script_;
  size_t last_found_;  // This is for an optimization used in LookupFilename.

  enum {
    kUninitialized,
    kReadScript,
    kNotReadScript,  // read of script failed.
  } state_;
};


// The implementation of TableWriter we use when writing directly
// to an archive plus an associated scp.
template<class Holder>
class TableWriterBothImpl: public TableWriterImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  virtual bool Open(const std::string &wspecifier) {
    switch (state_) {
      case kUninitialized:
        break;
      case kWriteError:
        KALDI_ERR << "TableWriter: opening stream, already open with write error.";
      case kOpen: default:
        if (!Close())  // throw because this error may not have been previously detected by user.
          KALDI_ERR << "TableWriter: opening stream, error closing previously open stream.";
    }
    wspecifier_ = wspecifier;
    WspecifierType ws = ClassifyWspecifier(wspecifier,
                                           &archive_wxfilename_,
                                           &script_wxfilename_,
                                           &opts_);
    KALDI_ASSERT(ws == kBothWspecifier);  // or wrongly called.
    if (ClassifyWxfilename(archive_wxfilename_) != kFileOutput)
      KALDI_WARN << "When writing to both archive and script, the script file "
          "will generally not be interpreted correctly unless the archive is "
          "an actual file: wspecifier = " << wspecifier;

    if (!archive_output_.Open(archive_wxfilename_, opts_.binary, false)) {  // false means no binary header.
      state_ = kUninitialized;
      return false;
    }
    if (!script_output_.Open(script_wxfilename_, false, false)) {  // first false means text mode:
      // script files always text-mode.   second false means don't write header (doesn't matter
      // for text mode).
      archive_output_.Close();  // Don't care about status: error anyway.
      state_ = kUninitialized;
      return false;
    }
    state_ = kOpen;
    return true;
  }

  virtual bool IsOpen() const {
    switch (state_) {
      case kUninitialized: return false;
      case kOpen: case kWriteError: return true;
      default: KALDI_ERR << "IsOpen() called on TableWriter in invalid state.";
    }
    return false;
  }

  void MakeFilename(typename std::ostream::pos_type streampos, std::string *output) const {
    std::ostringstream ss;
    ss << ':' << streampos;
    KALDI_ASSERT(ss.str() != ":-1");
    *output = archive_wxfilename_ + ss.str();
    
    // e.g. /some/file:12302.
    // Note that we warned if archive_wxfilename_ is not an actual filename;
    // the philosophy is we give the user rope and if they want to hang
    // themselves, with it, fine.
  }

  // Write returns true on success, false on failure, but
  // some errors may not be detected till we call Close().
  virtual bool Write(const std::string &key, const T &value) {
    switch (state_) {
      case kOpen: break;
      case kWriteError:
        // user should have known from the last
        // call to Write that there was a problem.  Warn about it.
        KALDI_WARN << "TableWriter: writing to non-open TableWriter object.";
        return false;
      case kUninitialized: default:
        KALDI_ERR << "TableWriter: Write called on invalid stream";
    }
    // state is now kOpen or kWriteError.
    if (!IsToken(key)) // e.g. empty string or has spaces...
      KALDI_ERR << "TableWriter: using invalid key " << key;
    std::ostream &archive_os = archive_output_.Stream();
    archive_os << key << ' ';
    typename std::ostream::pos_type archive_os_pos = archive_os.tellp();
    // position at start of Write() to archive.  We will record this in the script file.
    std::string offset_rxfilename;  // rxfilename with offset into the archive,
    // e.g. some_archive_name.ark:431541423
    MakeFilename(archive_os_pos, &offset_rxfilename);

    // Write to the script file first.
    // The idea is that we want to get all the information possible into the
    // script file, to make it easier to unwind errors later.
    std::ostream &script_os = script_output_.Stream();
    script_output_.Stream() << key << ' ' << offset_rxfilename << '\n';

    if (!Holder::Write(archive_output_.Stream(), opts_.binary, value)) {
      KALDI_WARN << "TableWriter: write failure to"
                 << PrintableWxfilename(archive_wxfilename_);
      state_ = kWriteError;
      return false;
    }

    if (script_os.fail()) {
      KALDI_WARN << "TableWriter: write failure to script file detected: "
                 << PrintableWxfilename(script_wxfilename_);
      state_ = kWriteError;
      return false;
    }

    if (archive_os.fail()) {
      KALDI_WARN << "TableWriter: write failure to archive file detected: "
                 << PrintableWxfilename(archive_wxfilename_);
      state_ = kWriteError;
      return false;
    }

    if (state_ == kWriteError) return false;  // Even if this Write seems to have
    // succeeded, we fail because a previous Write failed and the archive may be
    // corrupted and unreadable.

    if (opts_.flush)
      Flush();
    return true;
  }

  // Flush will flush any archive; it does not return error status,
  //  any errors will be reported on the next Write or Close.
  virtual void Flush() {
    switch (state_) {
      case kWriteError: case kOpen:
        archive_output_.Stream().flush();  // Don't check error status.
        script_output_.Stream().flush();  // Don't check error status.
        return;
      default:
        KALDI_WARN << "TableWriter: Flush called on not-open writer.";
    }
  }

  virtual bool Close() {
    if (!this->IsOpen())
      KALDI_ERR << "TableWriter: Close called on a stream that was not open.";
    bool close_success = true;
    if (archive_output_.IsOpen())
      if (!archive_output_.Close()) close_success = false;
    if (script_output_.IsOpen())
      if (!script_output_.Close()) close_success = false;
    bool ans = close_success && (state_ != kWriteError);
    state_ = kUninitialized;
    return ans;
  }

  TableWriterBothImpl(): state_(kUninitialized) {}

  // May throw on write error if Close() was not called.
  // User can get the error status by calling Close().
  virtual ~TableWriterBothImpl() {
    if (!IsOpen()) return;
    else if (!Close())
      KALDI_ERR << "At TableWriter destructor: Write failed or stream close failed: "
                << wspecifier_;
  }

 private:
  Output archive_output_;
  Output script_output_;
  WspecifierOptions opts_;
  std::string archive_wxfilename_;
  std::string script_wxfilename_;
  std::string wspecifier_;
  enum {               // is stream open?
    kUninitialized,    // no
    kOpen,             // yes
    kWriteError,       // yes
  } state_;
};


template<class Holder>
TableWriter<Holder>::TableWriter(const std::string &wspecifier): impl_(NULL) {
  if (wspecifier != "" && !Open(wspecifier)) {
    KALDI_ERR << "TableWriter: failed to write to "
              << wspecifier;
  }
}

template<class Holder>
bool TableWriter<Holder>::IsOpen() const {
  return (impl_ != NULL);
}


template<class Holder>
bool TableWriter<Holder>::Open(const std::string &wspecifier) {

  if (IsOpen()) {
    if (!Close()) // call Close() yourself to suppress this exception.
      KALDI_ERR << "TableWriter::Open, failed to close previously open writer.";
  }
  KALDI_ASSERT(impl_ == NULL);
  WspecifierType wtype = ClassifyWspecifier(wspecifier, NULL, NULL, NULL);
  switch (wtype) {
    case kBothWspecifier:
      impl_ = new TableWriterBothImpl<Holder>();
      break;
    case kArchiveWspecifier:
      impl_ = new TableWriterArchiveImpl<Holder>();
      break;
    case kScriptWspecifier:
      impl_ = new TableWriterScriptImpl<Holder>();
      break;
    case kNoWspecifier: default:
      KALDI_WARN << "ClassifyWspecifier: invalid wspecifier " << wspecifier;
      return false;
  }
  if (impl_->Open(wspecifier)) return true;
  else {  // The class will have printed a more specific warning.
    delete impl_;
    impl_ = NULL;
    return false;
  }
}

template<class Holder>
void TableWriter<Holder>::Write(const std::string &key,
                                const T &value) const {
  CheckImpl();
  if (!impl_->Write(key, value))
    KALDI_ERR << "Error in TableWriter::Write";
  // More specific warning will have
  // been printed in the Write function.
}

template<class Holder>
void TableWriter<Holder>::Flush() {
  CheckImpl();
  impl_->Flush();
}

template<class Holder>
bool TableWriter<Holder>::Close() {
  CheckImpl();
  bool ans = impl_->Close();
  delete impl_;  // We don't keep around non-open impl_ objects [c.f. definition of IsOpen()]
  impl_ = NULL;
  return ans;
}

template<class Holder>
TableWriter<Holder>::~TableWriter() {
  if (IsOpen() && !Close()) {
    KALDI_ERR << "Error closing TableWriter [in destructor].";
  }
}


// Types of RandomAccessTableReader:
// In principle, we would like to have four types of RandomAccessTableReader:
//  the 4 combinations  [scp, archive], [seekable, not-seekable],
// where if something is seekable we only store a file offset.  However,
// it seems sufficient for now to only implement two of these, in both
// cases assuming it's not seekable so we never store file offsets and always
// store either the scp line or the data in the archive.  The reasons are:
// (1)
// For scp files, storing the actual entry is not that much more expensive
// than storing the file offsets (since the entries are just filenames), and
// avoids a lot of fseek operations that might be expensive.
// (2)
// For archive files, there is no real reason, if you have the archive file
// on disk somewhere, why you wouldn't access it via its associated scp.
// [i.e. write it as ark, scp].  The main reason to read archives directly
// is if they are part of a pipe, and in this case it's not seekable, so
// we implement only this case.
//
// Note that we will rarely in practice have to keep in memory everything in
// the archive, as long as things are only read once from the archive (the
// "o, " or "once" option) and as long as we keep our keys in sorted order; to take
// advantage of this we need the "s, " (sorted) option, so we would read archives
// as e.g. "s, o, ark:-" (this is the rspecifier we would use if it was the
// standard input and these conditions held).

template<class Holder> class RandomAccessTableReaderImplBase {
 public:
  typedef typename Holder::T T;

  virtual bool Open(const std::string &rspecifier) = 0;

  virtual bool HasKey(const std::string &key) = 0;

  virtual const T &Value(const std::string &key) = 0;

  virtual bool Close() = 0;

  virtual ~RandomAccessTableReaderImplBase() {}
};


// Implementation of RandomAccessTableReader for a script file; for simplicity we
// just read it in all in one go, as it's unlikely someone would generate this
// from a pipe.  In principle we could read it on-demand as for the archives, but
// this would probably be overkill.

// Note: the code for this this class is similar to TableWriterScriptImpl:
// try to keep them in sync.
template<class Holder>
class RandomAccessTableReaderScriptImpl:
      public RandomAccessTableReaderImplBase<Holder> {

 public:
  typedef typename Holder::T T;

  RandomAccessTableReaderScriptImpl(): last_found_(0), state_(kUninitialized) {}

  virtual bool Open(const std::string &rspecifier) {
    switch (state_) {
      case kNotHaveObject: case kHaveObject: case kGaveObject:
        KALDI_ERR << " Opening already open RandomAccessTableReader: call Close first.";
      case kUninitialized: case kNotReadScript:
        break;
    }
    RspecifierType rs = ClassifyRspecifier(rspecifier,
                                           &script_rxfilename_,
                                           &opts_);
    KALDI_ASSERT(rs == kScriptRspecifier);  // or wrongly called.
    KALDI_ASSERT(script_.empty());  // no way it could be nonempty at this point.

    if (! ReadScriptFile(script_rxfilename_,
                        true,  // print any warnings
                        &script_)) {  // error reading script file or invalid format
      state_ = kNotReadScript;
      return false;  // no need to print further warnings.  user gets the error.
    }

    rspecifier_ = rspecifier;
    // If opts_.sorted, the user has asserted that the keys are already sorted.
    // Although we could easily sort them, we want to let the user know of this
    // mistake.  This same mistake could have serious effects if used with an
    // archive rather than a script.
    if (!opts_.sorted)
      std::sort(script_.begin(), script_.end());
    for (size_t i = 0; i+1 < script_.size(); i++) {
      if (script_[i].first.compare(script_[i+1].first) >= 0) {
        // script[i] not < script[i+1] in lexical order...
        bool same = (script_[i].first == script_[i+1].first);
        KALDI_WARN << "Script file " << PrintableRxfilename(script_rxfilename_)
                   << (same ? " contains duplicate key: " :
                       " is not sorted (remove s, option or add ns, option): key is ")
                   << script_[i].first;
        state_ = kNotReadScript;
        return false;
      }
    }
    state_ = kNotHaveObject;
    return true;
  }

  virtual bool IsOpen() const {
    return  (state_ == kNotHaveObject || state_ == kHaveObject ||
             state_ == kGaveObject);
  }

  virtual bool Close() {
    if (!IsOpen())
      KALDI_ERR << "Close() called on RandomAccessTableReader that was not open.";
    holder_.Clear();
    state_ = kUninitialized;
    last_found_ = 0;
    script_.clear();
    current_key_ = "";
    // This one cannot fail because any errors of a "global"
    // nature would have been detected when we did Open().
    // With archives it's different.
    return true;
  }

  virtual bool HasKey(const std::string &key) {
    bool preload = opts_.permissive;
    // In permissive mode, we have to check that we can read
    // the scp entry before we assert that the key is there.
    return HasKeyInternal(key, preload);
  }


  // Write returns true on success, false on failure, but
  // some errors may not be detected till we call Close().
  virtual const T&  Value(const std::string &key) {

    if (!IsOpen())
      KALDI_ERR << "RandomAccessTableReaderWriter: Value() called on "
          "not-open object.";

    if (!((state_ == kHaveObject || state_ == kGaveObject)
         && key == current_key_)) {  // Not already stored...
      bool has_key = HasKeyInternal(key, true);  // preload.
      if (!has_key)
        KALDI_ERR << "RandomAccessTableReader::Value(), could not get item for key "
                  << key << ", rspecifier is " << rspecifier_ << "[to ignore this "
                  << ", add the p, (permissive) option to the rspecifier.";
      KALDI_ASSERT(state_ == kHaveObject && key == current_key_);
    }

    if (state_ == kHaveObject) {
      state_ = kGaveObject;
      if (opts_.once) MakeTombstone(key);  // make sure that future lookups fail.
      return holder_.Value();
    } else {  // state_ == kGaveObject
      if (opts_.once)
        KALDI_ERR << "Value called twice for the same key\n";
      return holder_.Value();
    }
  }

  virtual ~RandomAccessTableReaderScriptImpl() {
    if (state_ == kHaveObject || state_ == kGaveObject)
      holder_.Clear();
  }

 private:
  // HasKeyInternal when called with preload == false just tells us whether
  // the key is in the scp.  With preload == true, it will also check that
  // we can preload the object from disk (loading from the rxfilename in
  // the scp), and only return true if we can.  This function is called
  // both from HasKey and from Value().
  virtual bool HasKeyInternal(const std::string &key, bool preload) {
    switch (state_) {
      case kUninitialized: case kNotReadScript:
        KALDI_ERR << "HasKey called on RandomAccessTableReader object that is not open.";
      default: break;
    }
    KALDI_ASSERT(IsToken(key));
    size_t key_pos = 0; // set to zero to suppress warning
    bool ans = LookupKey(key, &key_pos);
    if (!ans) return false;
    else {
      // First do a check regarding the "once" option.
      if (opts_.once && script_[key_pos].second == "") {  // A "tombstone"; user is asking about
        // already-read key.
        KALDI_ERR << "HasKey called on key whose value was already read, and "
            " you specified the \"once\" option (o, ): try removing o, or adding no, :"
            " rspecifier is " << rspecifier_;
      }
      if (!preload)
        return true;  // we have the key.
      else {  // preload specified, so we have to pre-load the object before returning true.
        if (!input_.Open(script_[key_pos].second)) {
          KALDI_WARN << "RandomAccessTableReader: error opening stream " << PrintableRxfilename(script_[key_pos].second);
          return false;
        } else {
          // Make sure holder empty.
          if (state_ == kHaveObject || state_ == kGaveObject)
            holder_.Clear();
          if (holder_.Read(input_.Stream())) {
            state_ = kHaveObject;
            current_key_ = key;
            return true;
          } else {
            KALDI_WARN << "RandomAccessTableReader: error reading object from "
                "stream " << PrintableRxfilename(script_[key_pos].second);
            state_ = kNotHaveObject;
            return false;
          }
        }
      }
    }
  }
  void MakeTombstone(const std::string &key) {
    size_t offset;
    if (!LookupKey(key, &offset))
      KALDI_ERR << "RandomAccessTableReader object in inconsistent state.";
    else
      script_[offset].second = "";
  }
  bool LookupKey(const std::string &key, size_t *script_offset) {
    // First, an optimization: if we're going consecutively, this will
    // make the lookup very fast.  Since we may call HasKey and then
    // Value(), which both may look up the key, we test if either the
    // current or next position are correct.
    if (last_found_ < script_.size() && script_[last_found_].first == key) {
      *script_offset = last_found_;
      return true;
    }
    last_found_++;
    if (last_found_ < script_.size() && script_[last_found_].first == key) {
      *script_offset = last_found_;
      return true;
    }
    std::pair<std::string, std::string> pr(key, "");  // Important that ""
    // compares less than or equal to any string, so lower_bound points to the
    // element that has the same key.
    typedef typename std::vector<std::pair<std::string, std::string> >::const_iterator 
        IterType;
    IterType iter = std::lower_bound(script_.begin(), script_.end(), pr);
    if (iter != script_.end() && iter->first == key) {
      last_found_ = *script_offset = iter - script_.begin();
      return true;
    } else {
      return false;
    }
  }


  Input input_;  // Use the same input_ object for reading each file, in case
  // the scp specifies offsets in an archive (so we can keep the same file open).
  RspecifierOptions opts_;
  std::string rspecifier_;  // rspecifier used to open it; used in debug messages
  std::string script_rxfilename_;  // filename of script.

  std::string current_key_;  // Key of object in holder_
  Holder holder_;

  // the script_ variable contains pairs of (key, filename), sorted using
  // std::sort.  This can be used with binary_search to look up filenames for
  // writing.  If this becomes inefficient we can use std::unordered_map (but I
  // suspect this wouldn't be significantly faster & would use more memory).
  // If memory becomes a problem here, the user should probably be passing
  // only the relevant part of the scp file rather than expecting us to get too
  // clever in the code.
  std::vector<std::pair<std::string, std::string> > script_;
  size_t last_found_;  // This is for an optimization used in FindFilename.

  enum {  //           [Do we have          [Does holder_
    //                script_ set up?]      contain object?]
    kUninitialized,  //     no                     no
    kNotReadScript,  //     no                     no
    kNotHaveObject,  //     yes                    no
    kHaveObject,   //     yes                    yes
    kGaveObject,   //     yes                    yes
    // [kGaveObject is as kHaveObject but we note that the
    //  user has already read it; this is for checking that
    // if "once" is specified, the user actually only reads
    // it once.
  } state_;

};




// This is the base-class (with some implemented functions) for the
// implementations of RandomAccessTableReader when it's an archive.  This
// base-class handles opening the files, storing the state of the reading
// process, and loading objects.  This is the only case in which we have
// an intermediate class in the hierarchy between the virtual ImplBase
// class and the actual Impl classes.
// The child classes vary in the assumptions regarding sorting, etc.

template<class Holder>  class RandomAccessTableReaderArchiveImplBase:
      public RandomAccessTableReaderImplBase<Holder> {
 public:
  typedef typename Holder::T T;

  RandomAccessTableReaderArchiveImplBase(): holder_(NULL), state_(kUninitialized) { }

  virtual bool Open(const std::string &rspecifier) {
    if (state_ != kUninitialized) {
      if (! this->Close()) // call Close() yourself to suppress this exception.
        KALDI_ERR << "TableReader::Open, error closing previous input.";
    }
    RspecifierType rs = ClassifyRspecifier(rspecifier, &archive_rxfilename_,
                                           &opts_);
    KALDI_ASSERT(rs == kArchiveRspecifier);

    // NULL means don't expect binary-mode header
    bool ans;
    if (Holder::IsReadInBinary())
      ans = input_.Open(archive_rxfilename_, NULL);
    else
      ans = input_.OpenTextMode(archive_rxfilename_);
    if (!ans) {  // header.
      KALDI_WARN << "TableReader: failed to open stream "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kUninitialized;  // Failure on Open
      return false;  // User should print the error message.
    } else {
      state_ = kNoObject;
    }
    return true;
  }

  // ReadNextObject() requires that the state be kNoObject,
  // and it will try read the next object.  If it succeeds,
  // it sets the state to kHaveObject, and
  // cur_key_ and holder_ have the key and value.  If it fails,
  // it sets the state to kError or kEof.
  void ReadNextObject() {
    if (state_ != kNoObject)
      KALDI_ERR << "TableReader: ReadNextObject() called from wrong state.";  // Code error
    // somewhere in this class or a child class.
    std::istream &is = input_.Stream();
    is.clear();  // Clear any fail bits that may have been set... just in case
    // this happened in the Read function.
    is >> cur_key_;  // This eats up any leading whitespace and gets the string.
    if (is.eof()) {
      state_ = kEof;
      return;
    }
    if (is.fail()) {  // This shouldn't really happen, barring file-system errors.
      KALDI_WARN << "Error reading archive "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      return;
    }
    int c;
    if ((c = is.peek()) != ' ' && c != '\t' && c != '\n') {  // We expect a space ' ' after the key.
      // We also allow tab, just so we can read archives generated by scripts that may
      // not be fully aware of how this format works.
      KALDI_WARN << "Invalid archive file format: expected space after key " <<cur_key_
                 <<", got character "
                 << CharToString(static_cast<char>(is.peek())) << ", reading "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      return;
    }
    if (c != '\n') is.get();  // Consume the space or tab.
    holder_ = new Holder;
    if (holder_->Read(is)) {
      state_ = kHaveObject;
      return;
    } else {
      KALDI_WARN << "Object read failed, reading archive "
                 << PrintableRxfilename(archive_rxfilename_);
      state_ = kError;
      delete holder_;
      holder_ = NULL;
      return;
    }
  }

  virtual bool IsOpen() const {
    switch (state_) {
      case kEof: case kError: case kHaveObject: case kNoObject: return true;
      case kUninitialized: return false;
      default: KALDI_ERR << "IsOpen() called on invalid object.";
        return false;
    }
  }

  // Called by the child-class virutal Close() functions; does the
  // shared parts of the cleanup.
  bool CloseInternal() {
    if (! this->IsOpen())
      KALDI_ERR << "Close() called on TableReader twice or otherwise wrongly.";
    if (input_.IsOpen())
      input_.Close();
    if (state_ == kHaveObject) {
      KALDI_ASSERT(holder_ != NULL);
      delete holder_;
      holder_ = NULL;
    } else KALDI_ASSERT(holder_ == NULL);
    bool ans = (state_ != kError);
    state_ = kUninitialized;
    if (!ans && opts_.permissive) {
      KALDI_WARN << "RandomAccessTableReader, error state detected closing reader.  "
                 << "Ignoring it because you specified permissive mode.";
      return true;
    }
    return ans;
  }

  ~RandomAccessTableReaderArchiveImplBase() {
    // The child class has the responsibility to call CloseInternal().
    KALDI_ASSERT(state_ == kUninitialized && holder_ == NULL);
  }
 private:
  Input input_;       // Input object for the archive
 protected:
  // The variables below are accessed by child classes.

  std::string cur_key_;   // current key (if state == kHaveObject).
  Holder *holder_;     // Holds the object we just read (if state == kHaveObject)

  std::string archive_rxfilename_;
  RspecifierOptions opts_;

  enum {  //  [The state of the reading process]               [does holder_ [is input_
    //                                                         have object]   open]
    kUninitialized,  // Uninitialized or closed                     no         no
    kNoObject,      // Do not have object in holder_               no         yes
    kHaveObject,    // Have object in holder_                      yes        yes
    kEof,           // End of file                                 no         yes
    kError,         // Some kind of error-state in the reading.    no         yes
  } state_;

};


// RandomAccessTableReaderDSortedArchiveImpl (DSorted for "doubly sorted") is the
// implementation for random-access reading of archives when both the archive,
// and the calling code, are in sorted order (i.e. we ask for the keys in sorted
// order).  This is when the s and cs options are both given.  It only ever has
// to keep one object in memory.  It inherits from
// RandomAccessTableReaderArchiveImplBase which implements the common parts of
// RandomAccessTableReader that are used when it's an archive we're reading from.

template<class Holder>  class RandomAccessTableReaderDSortedArchiveImpl:
      public RandomAccessTableReaderArchiveImplBase<Holder> {
  using RandomAccessTableReaderArchiveImplBase<Holder>::kUninitialized;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kHaveObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kNoObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kEof;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kError;
  using RandomAccessTableReaderArchiveImplBase<Holder>::state_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::opts_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::cur_key_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::holder_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::archive_rxfilename_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::ReadNextObject;
 public:
  typedef typename Holder::T T;

  RandomAccessTableReaderDSortedArchiveImpl() { }

  virtual bool Close() {
    // We don't have anything additional to clean up, so just
    // call generic base-class one.
    return this->CloseInternal();
  }

  virtual bool HasKey(const std::string &key) {
    return FindKeyInternal(key);
  }
  virtual const T & Value(const std::string &key) {
    if (FindKeyInternal(key)) {
      KALDI_ASSERT(this->state_ == kHaveObject && key == this->cur_key_
                   && holder_ != NULL);
      return this->holder_->Value();
    } else {
      KALDI_ERR << "RandomAccessTableReader: Value() called but no such key " << key
                << " in archive " << PrintableRxfilename(archive_rxfilename_);
      return *(const T*)NULL;  // keep compiler happy.
    }
  }

  virtual ~RandomAccessTableReaderDSortedArchiveImpl() {
    if (this->IsOpen())
      if (!Close()) // more specific warning will already have been printed.
        // we are in some kind of error state & user did not find out by
        // calling Close().
        KALDI_ERR << "Error closing RandomAccesseTableReader\n";
  }
 private:
  // FindKeyInternal tries to find the key by calling "ReadNextObject()"
  // as many times as necessary till we get to it.  It is called from
  // both FindKey and Value().
  bool FindKeyInternal(const std::string &key) {
    // First check that the user is calling us right: should be
    // in sorted order.  If not, error.
    if (!last_requested_key_.empty()) {
      if (key.compare(last_requested_key_) < 0) {  // key < last_requested_key_
        KALDI_ERR << "RandomAccessTableReader: you provided the \"cs\" option "
                  << "but are not calling with keys in sorted order: "
                  << key << " < " << last_requested_key_;
      }
    }
    // last_requested_key_ is just for debugging of order of calling.
    last_requested_key_ = key;

    if (state_ == kNoObject)
      ReadNextObject();  // This can only happen
      // once, the first time someone calls HasKey() or Value().  We don't
      // do it in the initializer to stop the program hanging too soon,
      // if reading from a pipe.

    if (state_ == kEof || state_ == kError) return false;

    if (state_ == kUninitialized)
      KALDI_ERR << "Trying to access a RandomAccessTableReader object that is not open.";

    std::string last_key_;  // To check that
    // the archive we're reading is in sorted order.
    while (1) {
      KALDI_ASSERT(state_ == kHaveObject);
      int compare = key.compare(cur_key_);
      if (compare == 0) {  // key == key_
        return true;  // we got it..
      } else if (compare < 0) {  // key < cur_key_, so we already read past the
        // place where we want to be.  This implies that we will never find it
        // [due to the sorting etc., this means it just isn't in the archive].
        return false;
      } else {  // compare > 0, key > cur_key_.  We need to read further ahead.
        last_key_ = cur_key_;
        // read next object.. we have to set state to kNoObject first.
        KALDI_ASSERT(holder_ != NULL);
        delete holder_;
        holder_ = NULL;
        state_ = kNoObject;
        ReadNextObject();
        if (state_ != kHaveObject)
          return false;  // eof or read error.
        if (cur_key_.compare(last_key_) <= 0) {
          KALDI_ERR << "RandomAccessTableReader: you provided the \"s\" option "
                    << " (sorted order), but keys are out of order or duplicated: "
                    << last_key_ << " is followed by " << cur_key_;
        }
      }
    }
  }

  /// Last string provided to HasKey() or Value();
  std::string last_requested_key_;


};

// RandomAccessTableReaderSortedArchiveImpl is for random-access reading of
// archives when the user specified the sorted (s) option but not the
// called-sorted (cs) options.
template<class Holder>  class RandomAccessTableReaderSortedArchiveImpl:
      public RandomAccessTableReaderArchiveImplBase<Holder> {
  using RandomAccessTableReaderArchiveImplBase<Holder>::kUninitialized;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kHaveObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kNoObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kEof;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kError;
  using RandomAccessTableReaderArchiveImplBase<Holder>::state_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::opts_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::cur_key_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::holder_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::archive_rxfilename_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::ReadNextObject;

 public:
  typedef typename Holder::T T;

  RandomAccessTableReaderSortedArchiveImpl():
      last_found_index_(static_cast<size_t>(-1)),
      pending_delete_(static_cast<size_t>(-1)) { }

  virtual bool Close() {
    for (size_t i = 0; i < seen_pairs_.size(); i++)
      if (seen_pairs_[i].second)
        delete seen_pairs_[i].second;
    seen_pairs_.clear();

    pending_delete_ = static_cast<size_t>(-1);
    last_found_index_ = static_cast<size_t>(-1);

    return this->CloseInternal();
  }
  virtual bool HasKey(const std::string &key) {
    HandlePendingDelete();
    size_t index;
    bool ans = FindKeyInternal(key, &index);
    if (ans && opts_.once && seen_pairs_[index].second == NULL) {
      // Just do a check RE the once option. "&&opts_.once" is for
      // efficiency since this can only happen in that case.
      KALDI_ERR << "Error: HasKey called after Value() already called for "
                << " that key, and once (o) option specified: "
                << PrintableRxfilename(archive_rxfilename_);
    }
    return ans;
  }
  virtual const T & Value(const std::string &key) {
    HandlePendingDelete();
    size_t index;
    if (FindKeyInternal(key, &index)) {
      if (seen_pairs_[index].second == NULL) {  // can happen if opts.once_
        KALDI_ERR << "Error: Value() called more than once for key "
                  << key << " and once (o) option specified: "
                  << PrintableRxfilename(archive_rxfilename_);
      }
      if (opts_.once)
        pending_delete_ = index;  // mark this index to be deleted on next call.
      return seen_pairs_[index].second->Value();
    } else {
      KALDI_ERR << "RandomAccessTableReader: Value() called but no such key " << key
                << " in archive " << PrintableRxfilename(archive_rxfilename_);
      return *(const T*)NULL;  // keep compiler happy.
    }
  }
  virtual ~RandomAccessTableReaderSortedArchiveImpl() {
    if (this->IsOpen())
      if (!Close()) // more specific warning will already have been printed.
        // we are in some kind of error state & user did not find out by
        // calling Close().
        KALDI_ERR << "Error closing RandomAccesseTableReader\n";
  }
 private:
  void HandlePendingDelete() {
    const size_t npos = static_cast<size_t>(-1);
    if (pending_delete_ != npos) {
      KALDI_ASSERT(pending_delete_ < seen_pairs_.size());
      KALDI_ASSERT(seen_pairs_[pending_delete_].second != NULL);
      delete seen_pairs_[pending_delete_].second;
      seen_pairs_[pending_delete_].second = NULL;
      pending_delete_ = npos;
    }
  }

  // FindKeyInternal tries to find the key in the array "seen_pairs_".
  // If it is not already there, it reads ahead as far as necessary
  // to determine whether we have the key or not.  On success it returns
  // true and puts the index into the array seen_pairs_, into "index";
  // on failure it returns false.
  // It will leave the state as either kNoObject, kEof or kError.
  // FindKeyInternal does not do any checking about whether you are asking
  // about a key that has been already given (with the "once" option).
  // That is the user's responsibility.

  bool FindKeyInternal(const std::string &key, size_t *index) {
    // First, an optimization in case the previous call was for the
    // same key, and we found it.
    if (last_found_index_ < seen_pairs_.size()
       && seen_pairs_[last_found_index_].first == key) {
      *index = last_found_index_;
      return true;
    }

    if (state_ == kUninitialized)
      KALDI_ERR << "Trying to access a RandomAccessTableReader object that is not open.";
    
    // Step one is to see whether we have to read ahead for the object..
    // Note, the possible states right now are kNoObject, kEof or kError.
    // We are never in the state kHaveObject except just after calling
    // ReadNextObject().
    bool looped = false;
    while (state_ == kNoObject &&
          (seen_pairs_.empty() || key.compare(seen_pairs_.back().first) > 0)) {
      looped = true;
      // Read this as:
      //  while ( the stream is potentially good for reading &&
      //        ([got no keys] || key > most_recent_key) ) { ...
      //     Try to read a new object.
      // Note that the keys in seen_pairs_ are ordered from least to greatest.
      ReadNextObject();
      if (state_ == kHaveObject) {  // Successfully read object.
        if (!seen_pairs_.empty() && // This is just a check.
           cur_key_.compare(seen_pairs_.back().first) <= 0) {
          // read the expression above as: !( cur_key_ > previous_key).
          // it means we are not in sorted order [the user specified that we
          // are, or we would not be using this implementation].
          KALDI_ERR << "RandomAccessTableReader: you provide the sorted (s) "
                    << "option but keys in archive "
                    << PrintableRxfilename(archive_rxfilename_) << " are not "
                    << "in sorted order: " << seen_pairs_.back().first
                    << " is followed by " << cur_key_;
        }
        KALDI_ASSERT(holder_ != NULL);
        seen_pairs_.push_back(std::make_pair(cur_key_, holder_));
        holder_ = NULL;
        state_ = kNoObject;
      }
    }
    if (looped) {  // We only need to check the last element of the seen_pairs_ array,
      // since we would not have read more after getting "key".
      if (!seen_pairs_.empty() && seen_pairs_.back().first == key) {
        last_found_index_ = *index = seen_pairs_.size() - 1;
        return true;
      } else return false;
    }
    // Now we have do an actual binary search in the seen_pairs_ array.
    std::pair<std::string, Holder*> pr(key, static_cast<Holder*>(NULL));
    typename std::vector<std::pair<std::string, Holder*> >::iterator
        iter = std::lower_bound(seen_pairs_.begin(), seen_pairs_.end(),
                                pr, PairCompare());
    if (iter != seen_pairs_.end() &&
       key == iter->first) {
      last_found_index_ = *index = (iter - seen_pairs_.begin());
      return true;
    } else return false;
  }

  // These are the pairs of (key, object) we have read.  We keep all the keys we
  // have read but the actual objects (if they are stored with pointers inside
  // the Holder object) may be deallocated if once == true, and the Holder
  // pointer set to NULL.
  std::vector<std::pair<std::string, Holder*> > seen_pairs_;
  size_t last_found_index_;  // An optimization s.t. if FindKeyInternal called twice with
  // same key (as it often will), it doesn't have to do the key search twice.
  size_t pending_delete_;  // If opts_.once == true, this is the index of
  // element of seen_pairs_ that is pending deletion.
  struct PairCompare {
    // PairCompare is the Less-than operator for the pairs of(key, Holder).
    // compares the keys.
    inline bool operator() (const std::pair<std::string, Holder*> &pr1,
                            const std::pair<std::string, Holder*> &pr2) {
      return  (pr1.first.compare(pr2.first) < 0);
    }
  };
};



// RandomAccessTableReaderUnsortedArchiveImpl is for random-access reading of
// archives when the user does not specify the sorted (s) option (in this case
// the called-sorted, or "cs" option, is ignored).  This is the least efficient
// of the random access archive readers, in general, but it can be as efficient
// as the others, in speed, memory and latency, if the "once" option is specified
// and it happens that the keys of the archive are the same as the keys the code
// is called with (to HasKey() and Value()), and in the same order.  However, if
// you ask it for a key that's not present it will have to read the archive till
// the end and store it all in memory.

template<class Holder>  class RandomAccessTableReaderUnsortedArchiveImpl:
      public RandomAccessTableReaderArchiveImplBase<Holder> {
  using RandomAccessTableReaderArchiveImplBase<Holder>::kUninitialized;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kHaveObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kNoObject;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kEof;
  using RandomAccessTableReaderArchiveImplBase<Holder>::kError;
  using RandomAccessTableReaderArchiveImplBase<Holder>::state_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::opts_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::cur_key_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::holder_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::archive_rxfilename_;
  using RandomAccessTableReaderArchiveImplBase<Holder>::ReadNextObject;

  typedef typename Holder::T T;

 public:
  RandomAccessTableReaderUnsortedArchiveImpl(): to_delete_iter_(map_.end()),
                                                to_delete_iter_valid_(false)
                                                 {
    map_.max_load_factor(0.5);  // make it quite empty -> quite efficient.
    // default seems to be 1.
  }

  virtual bool Close() {
    for (typename MapType::iterator iter = map_.begin();
        iter != map_.end();
        ++iter) {
      if (iter->second)
        delete iter->second;
    }
    map_.clear();
    first_deleted_string_ = "";
    to_delete_iter_valid_ = false;
    return this->CloseInternal();
  }

  virtual bool HasKey(const std::string &key) {
    HandlePendingDelete();
    return FindKeyInternal(key, NULL);
  }
  virtual const T & Value(const std::string &key) {
    HandlePendingDelete();
    const T *ans_ptr = NULL;
    if (FindKeyInternal(key, &ans_ptr))
      return *ans_ptr;
    else
      KALDI_ERR << "RandomAccessTableReader: Value() called but no such key " << key
                << " in archive " << PrintableRxfilename(archive_rxfilename_);
    return *(const T*)NULL;  // keep compiler happy.
  }
  virtual ~RandomAccessTableReaderUnsortedArchiveImpl() {
    if (this->IsOpen())
      if (!Close()) // more specific warning will already have been printed.
        // we are in some kind of error state & user did not find out by
        // calling Close().
        KALDI_ERR << "Error closing RandomAccesseTableReader\n";
  }
 private:
  void HandlePendingDelete() {
    if (to_delete_iter_valid_) {
      to_delete_iter_valid_ = false;
      delete to_delete_iter_->second;  // Delete Holder object.
      if (first_deleted_string_.length() == 0)
        first_deleted_string_ = to_delete_iter_->first;
      map_.erase(to_delete_iter_);  // delete that element.
    }
  }

  // FindKeyInternal tries to find the key in the map "map_"
  // If it is not already there, it reads ahead either until it finds the
  // key, or until end of file.  If called with value_ptr == NULL,
  // it assumes it's called from HasKey() and just returns true or false
  // and doesn't otherwise have side effects.  If called with value_ptr != 
  // NULL, it assumes it's called from Value().  Thus, it will crash
  // if it cannot find the key.  If it can find it it puts its address in
  // *value_ptr, and if opts_once == true it will mark that element of the
  // map to be deleted.

  bool FindKeyInternal(const std::string &key, const T **value_ptr = NULL) {
    typename MapType::iterator iter = map_.find(key);
    if (iter != map_.end()) {  // Found in the map...
      if (value_ptr == NULL) {  // called from HasKey
        return true;  // this is all we have to do.
      } else {
        *value_ptr = &(iter->second->Value());
        if (opts_.once) {  // value won't be needed again, so mark
          // for deletion.
          to_delete_iter_ = iter;  // pending delete.
          KALDI_ASSERT(!to_delete_iter_valid_);
          to_delete_iter_valid_ = true;
        }
        return true;
      }
    }
    while (state_ == kNoObject) {
      ReadNextObject();
      if (state_ == kHaveObject) {  // Successfully read object.
        state_ = kNoObject;  // we are about to transfer ownership
        // of the object in holder_ to map_.
        // Insert it into map_.
        std::pair<typename MapType::iterator, bool> pr =
            map_.insert(typename MapType::value_type(cur_key_, holder_));

        if (!pr.second) {  // Was not inserted-- previous element w/ same key
          delete holder_;  // map was not changed, no ownership transferred.
          KALDI_ERR << "Error in RandomAccessTableReader: duplicate key "
                    << cur_key_ << " in archive " << archive_rxfilename_;
        }
        holder_ = NULL;  // ownership transferred to map_.
        if (cur_key_ == key) {  // the one we wanted..
          if (value_ptr == NULL) {  // called from HasKey
            return true;
          } else {  // called from Value()
            *value_ptr = &(pr.first->second->Value());  // this gives us the
            // Value() from the Holder in the map.
            if (opts_.once) {  // mark for deletion, as won't be needed again.
              to_delete_iter_ = pr.first;
              KALDI_ASSERT(!to_delete_iter_valid_);
              to_delete_iter_valid_ = true;
            }
            return true;
          }
        }
      }
    }
    if (opts_.once && key == first_deleted_string_) {
      KALDI_ERR << "RandomAccessTableReader: you specified the once (o) option but "
          "you are calling using key " << key << " more than once: archive is "
                << PrintableRxfilename(archive_rxfilename_);
    }
    return false;  // We read the entire archive (or got to error state) and didn't
    // find it.
  }

  typedef std::unordered_map<std::string, Holder*, StringHasher>  MapType;
  MapType map_;

  typename MapType::iterator to_delete_iter_;
  bool to_delete_iter_valid_;

  std::string first_deleted_string_;  // keep the first string we deleted
  // from map_ (if opts_.once == true).  It's for an inexact spot-check that the
  // "once" option isn't being used incorrectly.

};





template<class Holder>
RandomAccessTableReader<Holder>::RandomAccessTableReader(const std::string &rspecifier):
    impl_(NULL) {
  if (rspecifier != "" && !Open(rspecifier))
    KALDI_ERR << "Error opening RandomAccessTableReader object "
        " (rspecifier is: " << rspecifier << ")";
}

template<class Holder>
bool RandomAccessTableReader<Holder>::Open(const std::string &rspecifier) {
  if (IsOpen())
    KALDI_ERR << "RandomAccessTableReader::Open(): already open.";
  RspecifierOptions opts;
  RspecifierType rs = ClassifyRspecifier(rspecifier, NULL, &opts);
  switch (rs) {
    case kScriptRspecifier:
      impl_ = new RandomAccessTableReaderScriptImpl<Holder>();
      break;
    case kArchiveRspecifier:
      if (opts.sorted) {
        if (opts.called_sorted) // "doubly" sorted case.
          impl_ = new RandomAccessTableReaderDSortedArchiveImpl<Holder>();
        else
          impl_ = new RandomAccessTableReaderSortedArchiveImpl<Holder>();
      } else impl_ = new RandomAccessTableReaderUnsortedArchiveImpl<Holder>();
      break;
    case kNoRspecifier: default:
      KALDI_WARN << "RandomAccessTableReader::Open(), invalid rspecifier: "
                 << rspecifier;
      return false;
  }
  if (impl_->Open(rspecifier))
    return true;
  else {
    // Warning will already have been printed.
    delete impl_;
    impl_ = NULL;
    return false;
  }
}

template<class Holder>
bool RandomAccessTableReader<Holder>::HasKey(const std::string &key) {
  CheckImpl();
  if (!IsToken(key))
    KALDI_ERR << "RandomAccessTableReader::HasKey, invalid key \"" << key << '"';
  return impl_->HasKey(key);
}


template<class Holder>
const typename RandomAccessTableReader<Holder>::T&
RandomAccessTableReader<Holder>::Value(const std::string &key) {
  CheckImpl();  
  return impl_->Value(key);
}

template<class Holder>
bool RandomAccessTableReader<Holder>::Close() {
  CheckImpl();
  bool ans =impl_->Close();
  delete impl_;
  impl_ = NULL;
  return ans;
}

template<class Holder>
RandomAccessTableReader<Holder>::~RandomAccessTableReader() {
  if (IsOpen() && !Close()) // call Close() yourself to stop this being thrown.
    KALDI_ERR << "RandomAccessTableReader destructor: failure detected.";
}

template<class Holder>
void SequentialTableReader<Holder>::CheckImpl() const {
  if (!impl_) {
    KALDI_ERR << "Trying to use empty SequentialTableReader (perhaps you "
              << "passed the empty string as an argument to a program?)";
  }
}

template<class Holder>
void RandomAccessTableReader<Holder>::CheckImpl() const {
  if (!impl_) {
    KALDI_ERR << "Trying to use empty RandomAccessTableReader (perhaps you "
              << "passed the empty string as an argument to a program?)";
  }
}

template<class Holder>
void TableWriter<Holder>::CheckImpl() const {
  if (!impl_) {
    KALDI_ERR << "Trying to use empty TableWriter (perhaps you "
              << "passed the empty string as an argument to a program?)";
  }
}

template<class Holder>
RandomAccessTableReaderMapped<Holder>::RandomAccessTableReaderMapped(
    const std::string &table_rxfilename,
    const std::string &utt2spk_rxfilename):
    reader_(table_rxfilename), token_reader_(table_rxfilename.empty() ? "" :
                                             utt2spk_rxfilename),
    utt2spk_rxfilename_(utt2spk_rxfilename) { }

template<class Holder>
bool RandomAccessTableReaderMapped<Holder>::Open(
    const std::string &table_rxfilename,
    const std::string &utt2spk_rxfilename) {
  if (reader_.IsOpen()) reader_.Close();
  if (token_reader_.IsOpen()) token_reader_.Close();
  KALDI_ASSERT(!table_rxfilename.empty());
  if (!reader_.Open(table_rxfilename)) return false; // will have printed
  // warning internally, probably.
  if (!utt2spk_rxfilename.empty()) {
    if (!token_reader_.Open(utt2spk_rxfilename)) {
      reader_.Close();
      return false;
    }
  }
  return true;
}


template<class Holder>
bool RandomAccessTableReaderMapped<Holder>::HasKey(const std::string &utt) {
  // We don't check IsOpen, we let the call go through to the member variable
  // (reader_), which will crash with a more informative error message than
  // we can give here, as we don't any longer know the rxfilename.
  if (token_reader_.IsOpen()) { // We need to map the key from utt to spk.
    if (!token_reader_.HasKey(utt))
      KALDI_ERR << "Attempting to read key " << utt << ", which is not present "
                << "in utt2spk map or similar map being read from "
                << PrintableRxfilename(utt2spk_rxfilename_);
    const std::string &spk = token_reader_.Value(utt);
    return reader_.HasKey(spk);
  } else {
    return reader_.HasKey(utt);
  }
}

template<class Holder>
const typename Holder::T& RandomAccessTableReaderMapped<Holder>::Value(
    const std::string &utt) {
  if (token_reader_.IsOpen()) { // We need to map the key from utt to spk.
    if (!token_reader_.HasKey(utt))
      KALDI_ERR << "Attempting to read key " << utt << ", which is not present "
                << "in utt2spk map or similar map being read from "
                << PrintableRxfilename(utt2spk_rxfilename_);
    const std::string &spk = token_reader_.Value(utt);
    return reader_.Value(spk);
  } else {
    return reader_.Value(utt);
  }
}



/// @}

} // end namespace kaldi



#endif
