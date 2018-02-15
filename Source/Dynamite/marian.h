// experimental emulator of Marian API on top of CNTK

#pragma once

#ifndef __MARIAN
#define __MARIAN

#define CNTK_BACKEND // we are running on the Marian-On-CntK Unified Platform

#include "CNTKLibrary.h"
#include "Models.h" // for the helper functions
#include "Layers.h" // for some functions missing in main CNTK, e.g. LogSoftmax
#include <vector>
#include <algorithm>
#include <memory>
#include <map>

#ifdef _MSC_VER
#ifndef __declspec_noreturn
#define __declspec_noreturn __declspec(noreturn)
#endif
#define __declspec_selectany __declspec(selectany)
#else
#ifndef __declspec_noreturn
#define __declspec_noreturn
#endif
#define __declspec_selectany __attribute__((weak))
#endif

template <typename... Args> static inline void ABORT_IF(bool cond, const char* msg, Args&&... /*ignoredArgs*/) { if (cond) CNTK::InvalidArgument("%s", msg); } // ignoring additional args for now
template <typename... Args> __declspec_noreturn static inline void ABORT(const char* msg, Args&&... /*ignoredArgs*/) { CNTK::InvalidArgument("%s", msg); } // ignoring additional args for now
// Marian is not warning-clean
#pragma warning(disable: 4267) // conversion from 'size_t' to 'int', possible loss of data
#pragma warning(disable: 4305) // truncation from 'double' to 'float'
#pragma warning(disable: 4100) // unreferenced formal parameter
#pragma warning(disable: 4099) // type name first seen using 'struct' now seen using 'class'
#pragma warning(disable: 4244) // conversion from 'int' to 'float', possible loss of data
#pragma warning(disable: 4189) // local variable is initialized but not referenced

#define YAML_REGISTER_TYPE(a, b) // not supported
#define PROJECT_VERSION_FULL "v1.0.0+d3fb526"

// needed for common/logging.h which is included by common/shape.h
#include "spdlog/spdlog.h"
namespace marian { class Config; }
void createLoggers(const marian::Config*) { }
std::shared_ptr<spdlog::logger> stderrLogger(const std::string&, const std::string&, const std::vector<std::string>&, bool quiet) { return spdlog::get(""); }

// these base-type headers can be used as-is
#include "common/shape.h" // Shape
#include "data/types.h"   // Word, Words, and related constants

namespace marian
{
    // -----------------------------------------------------------------------
    // basic types (Ptr, New, stuff, constants)
    // -----------------------------------------------------------------------

    template<typename T>
    using Ptr = std::shared_ptr<T>;
    template <class T, typename... Args> Ptr<T> New(Args&&... args) { return Ptr<T>(new T(std::forward<Args>(args)...)); }
    template <class T> Ptr<T> New(Ptr<T> p) { return Ptr<T>(p); }
    class ExpressionGraph;
    const float NEMATUS_LN_EPS = 1e-5f;
    typedef std::ofstream OutputFileStream;

    // -----------------------------------------------------------------------
    // Shape
    // -----------------------------------------------------------------------

    class ShapeProxy // Expr->shape() returns this, which performs direct translation without copying
    {
        const CNTK::NDShape& m_viewShape;
    public:
        ShapeProxy(const CNTK::NDShape& viewShape) : m_viewShape(viewShape) {}
        int operator[](int index) const // flips axis order, and interprets negative numbers
        {
            size_t rank = m_viewShape.Rank();
            if (index < 0)
                return m_viewShape[(size_t)(-(index + 1))];
            else
                return m_viewShape[rank - (size_t)(index + 1)];
        }
        const CNTK::NDShape& GetNDShape() const { return m_viewShape; }
        operator Shape() const // assigning to an actual vector object
        {
            size_t rank = m_viewShape.Rank();
            Shape shape; shape.resize(rank);
            for (size_t i = 0; i < rank; i++)
                shape.set((int)i, m_viewShape[rank - 1 - i]);
            return shape;
        }
        size_t size() const { return m_viewShape.Rank(); }
        int elements() const { return (int)m_viewShape.TotalSize(); }
        bool operator==(const ShapeProxy& other) const { return m_viewShape == other.m_viewShape; }
        bool operator!=(const ShapeProxy& other) const { return m_viewShape != other.m_viewShape; }
    };

    // -----------------------------------------------------------------------
    // Expr (<=> CNTK::Variable)
    // -----------------------------------------------------------------------

    class Expr : public CNTK::Variable
    {
        typedef CNTK::Variable Base;
#if 0
        void Trace() const
        {
            if (IsParameter())
                fprintf(stderr, "Parameter %S", Name().c_str());
            else if (IsConstant())
                fprintf(stderr, "Constant");
            else
            {
                let& owner = Owner();
                if (owner)
                {
                    fprintf(stderr, "%S^%u(", owner->OpName().c_str(), UniqueIdForDebugging());
                    let& args = owner->Inputs();
                    const char* delim = "";
                    for (let& arg : args)
                    {
                        fprintf(stderr, "%s%S", delim, arg.Shape().AsString().c_str());
                        delim = ", ";
                    }
                    fprintf(stderr, ")");
                }
                else
                    fprintf(stderr, "<unknown type>");
            }
            fprintf(stderr, " : %S\n", Shape().AsString().c_str()), fflush(stderr);
            //Value();   // this does not work presently because see-through ops on leaves are not allowed presently
        }
#else
        void Trace() const {}
#endif
    public:
        Expr(const CNTK::Variable& v) : Base(v)       { Trace(); }
        Expr(CNTK::Variable&& v) : Base(std::move(v)) { Trace(); }
        Expr(const CNTK::FunctionPtr& f) : Base(f)    { Trace(); }
        Expr(const std::nullptr_t&) : Base(CNTK::Variable()) { }
        Expr() : Base(CNTK::Variable()) { }
        Expr& operator=(const Expr& other) { Base::operator=(other); return *this; }
        // Marian accesses members by arrow, not dot. This is a bit of a hack.
        Expr* operator->() { return this; }
        const Expr* operator->() const { return this; }
        // Marian member functions
        CNTK::NDArrayViewPtr val() const { return Value(); }
        float scalar() const { return val()->AsScalar<float>(); }
        ShapeProxy shape() const { return Shape(); }
        void dump() const { Value()->LogToFile(Name()); }
        Ptr<ExpressionGraph> graph() const { return nullptr; } // TODO: We need a placeholder for this. Or maybe only allow one graph?
    };
    static inline void operator>> (const CNTK::NDArrayViewPtr& val, std::vector<float>& outputBuffer) { val->CopyDataTo(outputBuffer); }

    // -----------------------------------------------------------------------
    // helpers for mapping stuff to and from CNTK
    // -----------------------------------------------------------------------

    namespace mappers
    {
        static inline CNTK::NDShape ToNDShape(const Shape& shape)
        {
            // order of axes is reverse in Marian
            return CNTK::NDShape(shape.rbegin(), shape.rend());
        }
        static inline CNTK::Axis ToCNTKAxis(const Expr& x, int axisIndex)
        {
            const auto& viewShape = x.Shape();
            auto rank = viewShape.Rank();
            if (axisIndex < 0)
                axisIndex += (int)rank;
            if (axisIndex < 0 || axisIndex >= rank)
                CNTK::InvalidArgument("marian::ToCNTKAxis: axis out of range");
            return CNTK::Axis((int)rank - 1 - axisIndex);
        }
        template<typename Vector>
        static inline std::vector<CNTK::Axis> ToCNTKAxes(const Expr& x, const Vector/*collection<int>*/& axisIndices)
        {
            std::vector<CNTK::Axis> res(CNTK::Transform(axisIndices, [&](int axisIndex) { return ToCNTKAxis(x, axisIndex); }));
            std::reverse(res.begin(), res.end());
            return res;
        }
        template<typename Vector>
        static inline std::vector<CNTK::Variable> ToVariableVector(const Vector/*collection<Expr>*/& xs)
        {
            return std::vector<CNTK::Variable>(xs.begin(), xs.end());
            //return CNTK::TransformingSpan(xs, [&](const Expr& x) -> CNTK::Variable { return x; });
        }
    }

    // -----------------------------------------------------------------------
    // configuration (incl. keywords emulation)
    // -----------------------------------------------------------------------

    namespace keywords // to allow to say init=... etc
    {
        namespace Internal
        {
            template<typename T> struct KeywordPassThrough
            {
                const T& operator=(const T& other) { return other; } // just pass on the ref
            };
        };
#pragma push_macro("DEFINE_KEYWORD")
#define DEFINE_KEYWORD(type, name) typedef type name##_k; static Internal::KeywordPassThrough<type> name
        DEFINE_KEYWORD(CNTK::ParameterInitializer, init);
        DEFINE_KEYWORD(int,                        axis);
        DEFINE_KEYWORD(Expr,                       mask);
        DEFINE_KEYWORD(bool,                       fixed);
        DEFINE_KEYWORD(Shape,                      shape);
#pragma pop_macro("DEFINE_KEYWORD")
    };

    // helper to allow Options::get()'s return type depend on the template parameter
    template<class T> struct get_return                           { typedef const T& type; };
    template<>        struct get_return<std::string>              { typedef std::string type; };
    template<>        struct get_return<std::vector<int>>         { typedef std::vector<int> type; };
    template<>        struct get_return<std::vector<size_t>>      { typedef std::vector<size_t> type; };
    template<>        struct get_return<std::vector<std::string>> { typedef std::vector<std::string> type; };

    class Options : public CNTK::Dictionary
    {
        typedef CNTK::Dictionary Base;
        static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
        static std::string narrow(const std::wstring& s) { return std::string(s.begin(), s.end()); } // note: simplistic conversion that only works for 7-bit ASCII
    public:
        Options() {}
        Options(const CNTK::Dictionary& dict) : Base(dict) { }
        template<typename T>
        static std::vector<CNTK::DictionaryValue> VectorOf(const std::initializer_list<T>& v)
        {
            auto res = std::vector<CNTK::DictionaryValue>(CNTK::Transform(v, [](const T& val) -> DictionaryValue { return val; }));
            return res;
        }
        std::string str() { CNTK::LogicError("Option serialization not supported"); }
        void merge(const Ptr<Options>& other) // add all items from 'other' to 'this' unless an item already exists
        {
            for (const auto& key : other->Keys())
            {
                if (!Contains(key))
                    Base::operator[](key) = other->Base::operator[](key);
            }
        }
        template<typename T>
        void set(const std::string& key, const T& value);
        bool has(const std::string& key) const { return Base::Contains(widen(key)); }
        template<typename T>
        typename get_return<T>::type get(const std::string& key) const;
        template<typename T>
        typename get_return<T>::type get(const std::string& key, const T& deflt) const;
        const CNTK::DictionaryValue& operator[](const std::string& key) const { return Base::operator[](widen(key)); }
        const Options& getOptions() { return *this; }
    };

    // specializations must be done outside the class declaration
    template<typename T>
    inline void Options::set(const std::string& key, const T& value)
    {
        Base::operator[](widen(key)) = value;
    }
    template<>
    inline void Options::set<std::string>(const std::string& key, const std::string& value)
    {
        Base::operator[](widen(key)) = widen(value);
    }
    template<>
    inline void Options::set<const char*>(const std::string& key, const char* const& value)
    {
        Base::operator[](widen(key)) = widen(value);
    }
    template<typename T>
    inline typename get_return<T>::type Options::get(const std::string& key) const
    {
        return Base::operator[](widen(key)).Value<T>();
    }
    template<typename T>
    inline typename get_return<T>::type Options::get(const std::string& key, const T& deflt) const
    {
        return Base::GetOrElse(widen(key), deflt);
    }
    // Marian uses narrow strings  --inefficient, involves conversion and copy
    template<>
    inline typename get_return<std::string>::type Options::get<std::string>(const std::string& key) const
    {
        const auto& wstr = Base::operator[](widen(key)).Value<std::wstring>();
        return narrow(wstr);
    }
    template<>
    inline typename get_return<std::string>::type Options::get<std::string>(const std::string& key, const std::string& deflt) const
    {
        if (Base::Contains(widen(key))) // (inefficient due to double conversion, but keeps code simple)
            return get<std::string>(key);
        else
            return deflt;
    }
    template<> // vector<int>  --inefficient, involves conversion and copy
    inline typename get_return<std::vector<int>>::type Options::get<std::vector<int>>(const std::string& key) const
    {
        const auto& intArray = Base::operator[](widen(key)).Value<std::vector<CNTK::DictionaryValue>>(); // stored as an array of DictionaryValues, not ints
        return std::vector<int>(CNTK::Transform(intArray, [](const CNTK::DictionaryValue& v) { return v.Value<int>(); }));
    }
    template<>
    inline typename get_return<std::vector<size_t>>::type Options::get<std::vector<size_t>>(const std::string& key) const
    {
        const auto& intArray = Base::operator[](widen(key)).Value<std::vector<CNTK::DictionaryValue>>(); // stored as an array of DictionaryValues, not ints
        return std::vector<size_t>(CNTK::Transform(intArray, [](const CNTK::DictionaryValue& v) { return v.Value<size_t>(); }));
    }
    template<>
    inline typename get_return<std::vector<std::string>>::type Options::get<std::vector<std::string>>(const std::string& key) const
    {
        const auto& intArray = Base::operator[](widen(key)).Value<std::vector<CNTK::DictionaryValue>>(); // stored as an array of DictionaryValues, not ints
        return std::vector<std::string>(CNTK::Transform(intArray, [](const CNTK::DictionaryValue& v) { return narrow(v.Value<std::wstring>()); }));
    }

    struct Config
    {
        static size_t seed;
        class YamlNode : public Options // fake implementation used in place of YAML::Node. Operations supported.
        {
            struct ValueRef
            {
                template<typename T> void operator=(const T& value) { NOT_IMPLEMENTED; }; // dummy assignment operator (needed in load/save functions)
                ValueRef& operator[](const std::string&) { NOT_IMPLEMENTED; } // dummy index operator (needed in Amun)
            };
        public:
            ValueRef& operator[](const std::string&) { NOT_IMPLEMENTED; }
        };
        static void AddYamlToNpz(const YamlNode&, const std::string&, const std::string&) { NOT_IMPLEMENTED; }
    };
    __declspec_selectany size_t Config::seed;

    // -----------------------------------------------------------------------
    // data namespace
    // -----------------------------------------------------------------------

    namespace data
    {
        /*interface*/ class Batch // (this is a direct copy from batch.h)
        {
        public:
            virtual size_t size() const = 0;
            virtual size_t words() const { return 0; };
            virtual void debug() {};

            virtual std::vector<Ptr<Batch>> split(size_t n) = 0;

            const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }
            void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }

        protected:
            std::vector<size_t> sentenceIds_;
        };
        // SubBatchData -- return value of SubBatch::data()
        // This is a std::vector with special semantics of holding a one-hot tensor for CNTK.
        // This special type is recognized by rows(). This is the way to get the data tunneled in CNTK-internal format.
        class SubBatchData : public std::vector<Word>
        {
            typedef std::vector<Word> Base;
            Expr m_oneHot;     // CNTK only: alternatively, store as one-hot (same order as m_indices)
        public:
            SubBatchData() { }
            SubBatchData(int size, int width) : Base(size * width, 0) { }
            bool isOneHot() const { return !!m_oneHot; }
            const Expr& getOneHot() const { if (m_oneHot) return m_oneHot; else LogicError("SubBatchData::getOneHot() called when data is not one-hot"); }
            void setOneHot(Expr&& oneHot) { m_oneHot = std::move(oneHot); }
            const Base& getIndices() const { if (!m_oneHot) return *this; else LogicError("SubBatchData::getIndices() called when data is one-hot"); } //
            // type casts with dynamic runtime checks
            operator const Expr&() const { return getOneHot(); }
            operator const Base&() const { return getIndices(); }
        };
        // SubBatch - represents a batch of sentences of one data stream
        class SubBatch
        {
        public:
            SubBatch(int size, int width) : // create an empty batch (all mask values 0) of given dimensions
                m_indices(size, width), m_mask(size * width, 0), m_totalNumTokens(0),
                m_numSequences(size), m_maxSequenceLength(width)
            {
            }
            SubBatch(const std::vector<CNTK::Variable>& sequences) // CNTK-specific: alternatively encode CNTK batch in this data structure
            {
                // we just keep the data as a CNTK batch, but create the mask here already
                // Note that we keep a pointer to the CNTK batch, so the caller must keep the object alive.
                m_numSequences = sequences.size();
                m_maxSequenceLength = 0;
                m_totalNumTokens = 0;
                for (const auto& seq : sequences)
                {
                    size_t len = seq.size();
                    if (len > m_maxSequenceLength)
                        m_maxSequenceLength = len;
                    m_totalNumTokens += len;
                }
                // create the mask now, in regular interleaved Marian format
                m_mask.resize(m_numSequences * m_maxSequenceLength, 1); // init as if all words are present
                for (size_t s = 0; s < m_numSequences; s++) // now clear the mask flag for padding entries
                {
                    const auto& seq = sequences[s];
                    size_t len = seq.size();
                    for (size_t t = len; t < m_maxSequenceLength; t++)
                        m_mask[m_numSequences * t + s] = 0;
                }
                // create data tensor, in the same interleaved format
                // TODO: This way it is highly inefficient, and warrants a special kernel.
                //auto seqValues = std::vector<CNTK::NDArrayViewPtr>(CNTK::Transform(batch, [&](const CNTK::Variable& seq) { return seq.Value(); }));
                //auto batchValue = CNTK::Value::Create({ batch.front().Shape().Dimensions().front() }, seqValues, Dynamite::CurrentDevice());
                const auto& dummy = sequences.front()[0];
                std::vector<CNTK::Variable> interleavedWords(CNTK::Transform(CNTK::NumericRangeSpan<size_t>(m_numSequences * m_maxSequenceLength), [&](const size_t i)
                {
                    size_t t = i / m_numSequences;
                    size_t s = i % m_numSequences;
                    if (m_mask[i])
                        return sequences[s][t];
                    else
                        return dummy;
                }));
                //sequences[0].Value()->LogToFile(L"sent[0]");
                //sequences[1].Value()->LogToFile(L"sent[1]");
                CNTK::Variable interleavedBatch = CNTK::Splice(interleavedWords, CNTK::Axis(1));
                let vocabSize = dummy.Shape()[0];
                m_indices.setOneHot(CNTK::Reshape(interleavedBatch, { (size_t)vocabSize, m_numSequences, m_maxSequenceLength }));
                //m_oneHot.Value()->LogToFile(L"interleaved"), fflush(stderr);
            }
            int batchSize()  const { return (int)m_numSequences;  }
            int batchWidth() const { return (int)m_maxSequenceLength; }
            int batchWords() const { return (int)m_totalNumTokens; }
            void setWords(size_t words) { m_totalNumTokens = words; }
            SubBatchData& data() { return m_indices; }
            std::vector<float>& mask()   { return m_mask; }
            const SubBatchData& data() const { return m_indices; }
            const std::vector<float>& mask()   const { return m_mask; }
        private:
            size_t m_numSequences;       // number of sequences
            size_t m_maxSequenceLength;  // max sequence length
            size_t m_totalNumTokens;     // number of non-0 entries in m_mask
            // Sentence data is stored as a concatenation of all sequences, which have been padded
            // to m_maxSequenceLength. The resulting data can be reshaped to a column-major [T x S] tensor.
            // The mask is 1 for valid entries, and 0 for padding entries.
            SubBatchData m_indices;           // [positionInSequence * m_numSequences + sequenceIndex] word indices as interleaved batch like CNTK V1
            std::vector<float> m_mask;   // 1/0 mask of same size
        };
        class CorpusBatch : public Batch // represents a set of data streams, e.g. source and target
        {
        public:
            CorpusBatch() {}
            CorpusBatch(const std::vector<Ptr<SubBatch>>& streams) : m_streams(streams) {}
            CorpusBatch(std::vector<Ptr<SubBatch>>&& streams) : m_streams(std::move(streams)) {}
            size_t sets()  const { return m_streams.size(); }                                // get number of streams
            const Ptr<SubBatch>& operator[](size_t index) const { return m_streams[index]; } // get one stream
            const Ptr<SubBatch>& front() const { return m_streams.front(); }                 // get first stream (this would be source)
            const Ptr<SubBatch>& back()  const { return m_streams.back(); }                  // get last stream (this would be target)
            size_t size()  const { return front()->batchSize();  }                           // get number of sentences in first sub-batch
            size_t words() const { return front()->batchWords(); }                           // get #total present tokens in first stream (source)
            const std::vector<float>& getGuidedAlignment() { return m_guidedAlignment; }
            void setGuidedAlignment(const std::vector<float>& aln) { m_guidedAlignment = aln; }
            virtual std::vector<Ptr<Batch>> split(size_t n) override { n; CNTK::LogicError("CorpusBatch::split not implemented"); }
            std::vector<float>& getDataWeights() { NOT_IMPLEMENTED; }
            // helper for the initial run
            static Ptr<CorpusBatch> fakeBatch(const std::vector<size_t>& lengths, size_t batchSize, Ptr<Options> options)
            {
                auto batch = New<CorpusBatch>(std::vector<Ptr<SubBatch>>(CNTK::Transform(lengths, [&](size_t len)
                {
                    auto sb = New<SubBatch>((int)batchSize, (int)len);
                    size_t i = 0;
                    for (auto& v : sb->data())
                        v = i++;
                    std::fill(sb->mask().begin(), sb->mask().end(), 1.0f);
                    sb->setWords(sb->mask().size());
                    return sb;
                })));
                if (options->has("guided-alignment")) {
                    std::vector<float> alignment(batchSize * lengths.front() * lengths.back(),
                        0.f);
                    batch->setGuidedAlignment(alignment);
                }
                return batch;
            }
        private:
            std::vector<Ptr<SubBatch>> m_streams; // e.g. { source, target }
            std::vector<float> m_guidedAlignment; // [size() * front().batchWidth() * back().batchWidth()]
        };
        // dummy typedefs that are presently not supported
        struct Corpus {};
        struct BatchStats
        {
            void add(Ptr<data::CorpusBatch>, size_t = 1) { NOT_IMPLEMENTED; }
        };
    };

    // -----------------------------------------------------------------------
    // inits (initializers)
    // -----------------------------------------------------------------------

    namespace inits
    {
        typedef CNTK::ParameterInitializer ParameterInitializer;
        static CNTK::ParameterInitializer zeros = CNTK::ConstantInitializer(0);
        static CNTK::ParameterInitializer ones = CNTK::ConstantInitializer(1);
        static CNTK::ParameterInitializer glorot_uniform = CNTK::GlorotUniformInitializer(1.0); // TODO: check the scaling
        static inline CNTK::ParameterInitializer uniform() { return CNTK::UniformInitializer(0.1); }
        namespace Internal
        {
            static inline CNTK::ParameterInitializer WrappedVectorInitializer(const std::vector<float>& inputData)
            {
                return CNTK::Dictionary( // initializers are just dictionaries
                    L"from_vector",
                    // wrap the CPU-side buffer in an NDArrayView object (by pointer, no data is copied)
                    CNTK::NDArrayView(CNTK::DataType::Float, CNTK::NDShape{ inputData.size() },
                                      (void*)inputData.data(), inputData.size() * sizeof(float),
                                      CNTK::DeviceDescriptor::CPUDevice(), /*readOnly=*/true)
                );
            }
            template<typename T>
            static inline CNTK::ParameterInitializer CastVectorInitializer(const std::vector<T>& inputData)
            {
                // this version does make a copy, since a type cast is required
                CNTK::NDArrayView view(CNTK::DataType::Float, CNTK::StorageFormat::Dense, CNTK::NDShape{ inputData.size() }, CNTK::DeviceDescriptor::CPUDevice());
                auto* p = view.WritableDataBuffer<float>();
                for (auto v : inputData)
                    *p++ = (float)v;
                return CNTK::Dictionary(L"from_vector", std::move(view));
            }
            struct NumpyObject // only enough interface to get Amun to compile
            {
                std::vector<size_t> shape;
            };
        };
        template<typename T>
        static inline CNTK::ParameterInitializer from_vector(const std::vector<T>&     inputData) { return Internal::CastVectorInitializer(inputData); }
        static inline CNTK::ParameterInitializer from_vector(const std::vector<float>& inputData) { return Internal::WrappedVectorInitializer(inputData); }
        static inline CNTK::ParameterInitializer from_value(float value) { return CNTK::ConstantInitializer(value); }
        static inline CNTK::ParameterInitializer from_word2vec(const std::string& file, int dimVoc, int dimEmb, bool normalize = false) { file, dimVoc, dimEmb, normalize; CNTK::LogicError("from_word2vec: not implemented"); }
        static inline CNTK::ParameterInitializer from_numpy(const Internal::NumpyObject&) { NOT_IMPLEMENTED; }
    }

    // -----------------------------------------------------------------------
    // ops
    // Most are direct mappings onto the corresponding CNTK operations.
    // -----------------------------------------------------------------------

    namespace InternalOps // helper functions that are not actually part of the Marian API
    {
        static inline Expr NotImplemented(const char* s) { CNTK::LogicError("%s", s); return CNTK::Variable(); }

        // helper to implement sum over many elements
        static inline Expr plus(const std::vector<Expr>::const_iterator& b, const std::vector<Expr>::const_iterator& e) // TODO: use fused Gather/ReduceSum?
        {
            size_t n = e - b;
            auto mid = b + n / 2;
            if (mid == b)
                return *b;
            else
                return CNTK::Plus(plus(b, mid), plus(mid, e));
        }

        // constants
        // TODO: for adding and multiplying scalars, we should have special operations, without creating Constants all the time
        // this is not efficient, but all we can do presently
        template<typename Number>
        static inline Expr Scalar(Number x) { return (CNTK::Variable)CNTK::Constant::Scalar(CNTK::DataType::Float, (float)x, Dynamite::CurrentDevice()); }
        static inline Expr Constant(const CNTK::NDShape& viewShape, const CNTK::ParameterInitializer& init, bool isVolatile = false)
        {
            if (init.Contains(L"from_vector"))
            {
                // BUGBUG: This keeps a reference to the vector, not a copy, which only works inside a single expression, if at all.
                const auto& initData = init[L"from_vector"].Value<CNTK::NDArrayView>();
                if (initData.Shape().TotalSize() != viewShape.TotalSize())
                    CNTK::InvalidArgument("marian::constant: vector size does not match viewShape");
                // copy the supplied CPU buffer, which may be a temporary, to a GPU-side NDArrayView
                return (CNTK::Variable)CNTK::Constant(initData.AsShape(viewShape)->DeepClone(Dynamite::CurrentDevice(), /*readOnly=*/true), isVolatile);
            }
            CNTK::InvalidArgument("BUGBUG: no public Constant() from ParameterInitializer?");
        }
        static inline Expr Constant(const Shape& npShape, const CNTK::ParameterInitializer& init, bool isVolatile = false)
        {
            auto viewShape = mappers::ToNDShape(npShape); // convert to CNTK's column-major viewShape
            return Constant(viewShape, init, isVolatile);
        }

        // dropout
#if 0
        static inline std::vector<float> DropoutMask(size_t n, float dropProb)
        {
            // PERF BUGBUG: For now, we determine the dropout mask on the CPU. Instead, we should get the rand op to work under Dynamite.
            static int seed = 1;
            srand(seed++);
            float keepProb = 1 - dropProb;
            float RAND_MAXxP = keepProb * RAND_MAX;
            float invKeepProb = 1 / keepProb;
            std::vector<float> mask(CNTK::Transform(CNTK::NumericRangeSpan<size_t>(n), [&](size_t)
            {
                return (rand() < RAND_MAXxP) ? invKeepProb : 0;
            }));
            return mask;
        }
#endif
        static inline Expr DropoutMask(double dropRate, const CNTK::NDShape& shape)
        {
            // TODO: Where to store the state? Note that Marian uses a static in the back end.
            static CNTK::RNGState rngState;
            let keepRate = 1 - dropRate;
            return CNTK::BernoulliRandom(CNTK::NDArrayView::LazilyCreateRNGState(rngState, /*seed=*/ CNTK::SentinelValueForAutoSelectRandomSeed, Dynamite::CurrentDevice()),
                                         shape, Dynamite::CurrentDataType(), /*mean=*/keepRate, /*scale=*/1 / keepRate);
        }

        // Reshape helper
        static inline Expr Reshape(const CNTK::Variable& operand, const CNTK::NDShape& newShape)
        {
            if (!operand.IsConstant())
                return CNTK::Reshape(operand, newShape);
            // special treatment of constants: recreate a new constant with the correct shape
            // This is because AutoBatch can currently Functions, but not Constants, but Marian reshapes its inputs.
            // If we change AutoBatch's redirect from redirecting Functions to redirecting Variables, this workaround can be removed.
            const auto& val = operand.Value();
            auto valReshaped = val->AsShape(newShape);
            return (CNTK::Variable)CNTK::Constant(valReshaped);
        }
    };

    static inline Expr operator-(const Expr& a) { return CNTK::Negate(a); }

    static inline Expr operator+(const Expr& a, const Expr& b) { return CNTK::Plus(a, b); }
    static inline Expr operator-(const Expr& a, const Expr& b) { return CNTK::Minus(a, b); }
    static inline Expr operator*(const Expr& a, const Expr& b) { return CNTK::ElementTimes(a, b); }
    static inline Expr operator/(const Expr& a, const Expr& b) { return CNTK::ElementDivide(a, b); }

    static inline Expr operator+(double a, const Expr& b) { return a == 0 ? b : (Expr)CNTK::ScaleAndShift(b, /*scale=*/1, /*shift=*/a); }
    static inline Expr operator+(const Expr& a, double b) { return b == 0 ? a : (Expr)CNTK::ScaleAndShift(a, /*scale=*/1, /*shift=*/b); }

    static inline Expr operator-(double a, const Expr& b) { return a == 0 ? -b : (Expr)CNTK::ScaleAndShift(b, /*scale=*/-1, /*shift=*/ a); }
    static inline Expr operator-(const Expr& a, double b) { return b == 0 ?  a : (Expr)CNTK::ScaleAndShift(a, /*scale=*/ 1, /*shift=*/-b); }

    static inline Expr operator*(double a, const Expr& b) { return a == 1 ? b : (Expr)CNTK::ScaleAndShift(b, /*scale=*/a, /*shift=*/0); }
    static inline Expr operator*(const Expr& a, double b) { return b == 1 ? a : (Expr)CNTK::ScaleAndShift(a, /*scale=*/b, /*shift=*/0); }

    static inline Expr operator/(double a, const Expr& b) { return a == 1 ? (Expr)CNTK::Reciprocal(b) : InternalOps::Scalar(a) / b; } // TODO: watch out for our weird Reciprocal definition
    static inline Expr operator/(const Expr& a, double b) { return b == 1 ? a : (Expr)CNTK::ScaleAndShift(a, /*scale=*/1.0 / b, /*shift=*/0); }

    // these explicit overloads seem needed to avoid compiler confusion
    static inline Expr operator+(float  a, const Expr& b) { return (double)a + b; }
    static inline Expr operator+(int    a, const Expr& b) { return (double)a + b; }
    static inline Expr operator+(const Expr& a, float  b) { return a + (double)b; }
    static inline Expr operator+(const Expr& a, int    b) { return a + (double)b; }

    static inline Expr operator-(float  a, const Expr& b) { return (double)a - b; }
    static inline Expr operator-(int    a, const Expr& b) { return (double)a - b; }
    static inline Expr operator-(const Expr& a, float  b) { return a - (double)b; }
    static inline Expr operator-(const Expr& a, int    b) { return a - (double)b; }

    static inline Expr operator*(float  a, const Expr& b) { return (double)a * b; }
    static inline Expr operator*(int    a, const Expr& b) { return (double)a * b; }
    static inline Expr operator*(const Expr& a, float  b) { return a * (double)b; }
    static inline Expr operator*(const Expr& a, int    b) { return a * (double)b; }

    static inline Expr operator/(float  a, const Expr& b) { return (double)a / b; }
    static inline Expr operator/(int    a, const Expr& b) { return (double)a / b; }
    static inline Expr operator/(const Expr& a, float  b) { return a / (double)b; }
    static inline Expr operator/(const Expr& a, int    b) { return a / (double)b; }

    static inline Expr debug(const Expr& a, const std::string& message = "") { message; return a; } // not implemented presently

    static inline Expr plus(const std::vector<Expr>& xs) { return InternalOps::plus(xs.begin(), xs.end()); }

    static inline Expr logit(const Expr& a) { return CNTK::Sigmoid(a); }
    static inline Expr logit(const std::vector<Expr>& xs) { return logit(plus(xs)); }

    static inline Expr swish(const Expr& a) { return a * CNTK::Sigmoid(a); }
    static inline Expr swish(const std::vector<Expr>& xs) { return swish(plus(xs)); }

    static inline Expr tanh(const std::vector<Expr>& xs) { return CNTK::Tanh(plus(xs)); }
    template <typename... Args>
    static inline Expr tanh(Args... args) {
        std::vector<Expr> nodes{ args... };
        return tanh(nodes);
    }
    static inline Expr tanh(const Expr& x) { return CNTK::Tanh(x, L"Tanh(" + x.Name() + L")"); }

    static inline Expr relu(const Expr& a) { return CNTK::ReLU(a); }
    static inline Expr relu(const std::vector<Expr>& xs) { return relu(plus(xs)); }

    static inline Expr leakyrelu(const Expr& a) { a; return InternalOps::NotImplemented("leakyrelu"); }
    static inline Expr leakyrelu(const std::vector<Expr>& xs) { return leakyrelu(plus(xs)); }

    static inline Expr prelu(const Expr& a, float alpha = 0.01) { a, alpha; return InternalOps::NotImplemented("prelu"); }
    static inline Expr prelu(const std::vector<Expr>& xs, float alpha = 0.01) { return prelu(plus(xs), alpha); }

    static inline Expr log(const Expr& a) { return CNTK::Log(a); }

    static inline Expr exp(const Expr& a) { return CNTK::Exp(a); }

    // Expr pow(const Expr& a, Expr b);
    // Expr pow(float a, Expr b);
    // Expr pow(const Expr& a, float b);

    static inline Expr dot(const Expr& a, const Expr& b, bool transA = false, bool transB = false, float scalar = 1.f)
    {
        a, b, transA, transB, scalar;
        return InternalOps::NotImplemented("dot");
    }
    static inline Expr bdot(const Expr& aMarian, const Expr& bMarian, bool transAMarian = false, bool transBMarian = false, float scalar = 1.f)
    {
        // We operate on the internal CNTK notation which has all axes reversed in order.
        // For matrix products, we also need to swap the argument order.
        // TODO: The transposition could be free, but currently is not. At least pass an optimization hint that it does not need to make it dense?
        auto transA = transBMarian;
        auto transB = transAMarian;
        const auto& a = !transA ? bMarian : (Expr)TransposeAxes(bMarian, CNTK::Axis(0), CNTK::Axis(1));
        const auto& b = !transB ? aMarian : (Expr)TransposeAxes(aMarian, CNTK::Axis(0), CNTK::Axis(1));
        // This version is hard-coded for 2+ dimensions.
        const auto& aShape = a.Shape();
        const auto& bShape = b.Shape();
        size_t rank = aShape.Rank();
        if (rank < 2 || rank != bShape.Rank())
            CNTK::InvalidArgument("bdot: batch dimension(s) must match between arguments");
        const auto& aDims = aShape.Dimensions();
        const auto& bDims = bShape.Dimensions();
        auto I  = aDims[0];
        auto J  = aDims[1];
        auto J2 = bDims[0];
        auto K  = bDims[1];
        if (J != J2)
            CNTK::InvalidArgument("bdot: inner matrix dimensions must match between arguments");
        auto mapDimsBegin = aDims.begin() + 2; // Marian semantics is hard-coded strictly for matrices (2D tensors)
        auto mapDimsEnd   = aDims.end();
        // The matrix product is implemented as an InnerProduct after Reshape.
        //   a[I x J x B...] * b[J x K x B...] gets rewritten as:
        // = a[I x J x 1 x B...]
        // * b[1 x J x K x B...]
        // >> sum ^^^
        // ->c[I x 1 x K x B...
        // >> drop^^^
        // insert the axis
        CNTK::NDShapeDimensions aShapeExtended(rank + 1, 0);
        aShapeExtended[0] = I;
        aShapeExtended[1] = J;
        aShapeExtended[2] = 1;
        std::copy(mapDimsBegin, mapDimsEnd, aShapeExtended.begin() + 3);
        CNTK::NDShapeDimensions bShapeExtended = aShapeExtended;
        bShapeExtended[0] = 1;
        bShapeExtended[1] = J;
        bShapeExtended[2] = K;
        auto aExtended = InternalOps::Reshape(a, CNTK::NDShape(std::move(aShapeExtended))); // this is free in Dynamite
        auto bExtended = InternalOps::Reshape(b, CNTK::NDShape(std::move(bShapeExtended)));
        // perform the matrix product
        auto cExtended = CNTK::InnerProduct(aExtended, bExtended, CNTK::Axis(1));
        // remove the extra axis
        CNTK::NDShapeDimensions cShape(rank, 0);
        cShape[0] = I;
        cShape[1] = K;
        std::copy(mapDimsBegin, mapDimsEnd, cShape.begin() + 2);
        auto c = CNTK::Reshape(cExtended, CNTK::NDShape(std::move(cShape)));
        // final scaling
        if (scalar != 1)
            c = c * scalar;
        return c;
    }

    static inline Expr transpose(const Expr& a) { return CNTK::TransposeAxes(a, CNTK::Axis(0), CNTK::Axis(1)); }
    static inline Expr transpose(const Expr& a, const std::vector<int>& axes) { return CNTK::Transpose(a, mappers::ToCNTKAxes(a, axes)); }

    static inline Expr concatenate(const std::vector<Expr>& concats, keywords::axis_k ax = 0)
    {
        return CNTK::Splice(mappers::ToVariableVector(concats), mappers::ToCNTKAxis(concats.front(), ax));
    }
    static inline Expr repeat(const Expr& a, size_t repeats, keywords::axis_k ax = 0)
    {
        // TODO: This is not efficient. We just need to broadcast into a new axis, then Reshape, but there is no CNTK op that allows tat.
        if (repeats == 1)
            return a;
        return concatenate(std::vector<Expr>(repeats, a), ax);
    }

    static inline Expr reshape(const Expr& a, Shape ndShape) { return InternalOps::Reshape(a, mappers::ToNDShape(ndShape)); }

    static inline Expr atleast_nd(const Expr& a, size_t dims)
    {
        const auto& viewShape = a.Shape();
        if (viewShape.Rank() >= dims)
            return a;
        else
            return InternalOps::Reshape(a, viewShape.AppendAxis(dims -1, 1)); // pad with ones at end
    }
    static inline Expr atleast_1d(const Expr& a) { return atleast_nd(a, 1); }
    static inline Expr atleast_2d(const Expr& a) { return atleast_nd(a, 2); }
    static inline Expr atleast_3d(const Expr& a) { return atleast_nd(a, 3); }
    static inline Expr atleast_4d(const Expr& a) { return atleast_nd(a, 4); }

    static inline Expr flatten(const Expr& a) { return InternalOps::Reshape(a, { a.Shape().TotalSize() }); }
    static inline Expr flatten_2d(const Expr& a)
    {
        const auto& viewShape = a.Shape();
        size_t I = viewShape.Dimensions().front();
        size_t J = viewShape.TotalSize() / I; // all except first NDShape axis get flattened
        return InternalOps::Reshape(a, { I, J });
    }

    static inline Expr rows(const Expr& a, const std::vector<size_t>& indices)
    {
        const auto& viewShape = a.Shape();
        if (viewShape.Rank() != 2)
            CNTK::InvalidArgument("rows: data must be a matrix");
        size_t numClasses = viewShape.Dimensions().back();
        std::vector<float> indicesFloat(CNTK::Transform(indices, [](size_t index) { return (float)index; }));
        auto indicesVar = InternalOps::Constant(CNTK::NDShape{ indices.size() }, inits::Internal::WrappedVectorInitializer(indicesFloat));
        auto indicesOneHot = CNTK::OneHotOp(indicesVar, numClasses, /*outputSparse=*/true, CNTK::Axis(0));
        return CNTK::Times(a, indicesOneHot);
    }
    static inline Expr cols(const Expr& a, const std::vector<size_t>& indices) { return CNTK::Transpose(rows(CNTK::Transpose(a), indices)); } // note: not efficient
    // CNTK only:
    static inline Expr rows(const Expr& a, const Expr& oneHot)
    {
        return Times(a, oneHot);
    }
    // Marian-on-CNTK's SubBatch::data() returns an SubBatchData type, which may contain either a vector or a one-hot Expr.
    // We catch this here and dispatch to the respective function.
    static inline Expr rows(const Expr& a, const data::SubBatchData& indices)
    {
        if (indices.isOneHot())
            return rows(a, indices.getOneHot());
        else
            return rows(a, indices.getIndices());
    }

    static inline Expr select(const Expr& a, int axis, const std::vector<size_t>& indices)
    {
        a, axis, indices; // TODO: find out the semantics
        return InternalOps::NotImplemented("select");
    }

    static inline Expr sum(const Expr& a, keywords::axis_k ax = 0) { return CNTK::ReduceSum(a, mappers::ToCNTKAxis(a, ax)); }

    static inline Expr softmax(const Expr& a) { return Dynamite::Softmax(a, CNTK::Axis(0)); }
    static inline Expr softmax(const Expr& a, Expr mask)
    {
        a, mask; // TODO: find out the semantics
        return InternalOps::NotImplemented("softmax");
    }

    static inline Expr logsoftmax(const Expr& x) { return Dynamite::LogSoftmax(x, CNTK::Axis(0), L"LogSoftmax(" + x.Name() + L",Axis(0))"); }

    static inline Expr mean(const Expr& a, keywords::axis_k ax = 0) { return CNTK::ReduceMean(a, mappers::ToCNTKAxis(a, ax)); }

    // o = unnormalized log prob; y = label as an index if dense, and CNTK one-hot if sparse
    static inline Expr cross_entropy(const Expr& o, const Expr& y)
    {
        auto numClasses = o.Shape()[0];
        if (y.IsSparse())
            return Dynamite::CrossEntropyWithSoftmax(o, y, CNTK::Axis(0));
        else
            return Dynamite::CrossEntropyWithSoftmax(o, CNTK::OneHotOp(y, numClasses, /*outputSparse=*/true, CNTK::Axis(0)), CNTK::Axis(0));
    }

    static inline Expr affine(const Expr& x, const Expr& W, const Expr& b, bool transX = false, bool transW = false, float scalar = 1.0f)
    {
        auto res =
            /*if*/ (transW) ?
                CNTK::TransposeAffine(W, transX ? Transpose(x) : x, b)
            /*else*/ :
                CNTK::Affine(W, transX ? Transpose(x) : x, b);
        if (scalar != 1.0f)
            res = res * scalar;
        return res;
    }

    static inline Expr scalar_product(const Expr& a, Expr b, keywords::axis_k ax = 0) { return CNTK::InnerProduct(a, b, mappers::ToCNTKAxis(a, ax)); }

    static inline Expr weighted_average(const Expr& in, Expr weights, keywords::axis_k ax = 0)
    {
        Expr numer = CNTK::ReduceSum(in * weights, mappers::ToCNTKAxis(in, ax));
        Expr denom = CNTK::ReduceSum(     weights, mappers::ToCNTKAxis(in, ax));
        return numer / denom;
    }

    static inline Expr step(const Expr& a, int step, int ax)
    {
        // TODO: can 'step' be negative as well?
        return Slice(a, mappers::ToCNTKAxis(a, ax), step, step + 1);
    }

    static inline Expr sqrt(const Expr& a, float eps = 0.f) { return CNTK::Sqrt(a + eps); }
    static inline Expr square(const Expr& a) { return a * a; }

    static inline Expr layer_norm(const Expr& x, Expr gamma, Expr beta = nullptr, float eps = 1e-9)
    {
        // TODO: Expr == nullptr must be implemented. Variable::IsValid()?
        // Marian LayerNorm normalizes over Axis(0).
        auto axis = CNTK::Axis(0);
        Expr mean = ReduceMean(x, axis);
        Expr x0 = x - mean;
        auto var = InnerProduct(x0, x0, axis) / (float)x0.Shape().Dimensions().front();
        //auto invStdDev = 1 / Sqrt(var / (float)x0.Shape().Dimensions().front() + eps);
        auto invStdDev = Pow(var + eps, InternalOps::Scalar(-0.5));
        return CNTK::NormalizeDenormalize(x, mean, invStdDev, gamma, beta);
    }

    static inline Expr highway(const Expr& y, Expr x, Expr t)
    {
        y, x, t; // TODO: find out the semantics
        return InternalOps::NotImplemented("highway");
    }
    static inline Expr highway(const std::string prefix, Expr x)
    {
        prefix, x; // TODO: find out the semantics w.r.t. prefix
        return InternalOps::NotImplemented("highway");
    }

    static inline Expr dropout(const Expr& x, const Expr& mask) { return x * mask; }
    static inline Expr dropout(const Expr& x, float dropRate)
    {
        // untested. Check the dimension stuff.
        auto mask = InternalOps::DropoutMask(dropRate, x->Shape());
        return dropout(x, mask);
    }

    static inline Expr shift(const Expr& x, const Shape& npShape)
    {
        // BUGBUG: Complete this. Marian cannot take views, so this is unnecessarily complex.
        // The cheapest would be a convolution with a kernel shifted by 1.
        int mustBeDim = 1;
        for (auto dim : npShape)
        {
            if (dim != mustBeDim)
                CNTK::InvalidArgument("marian::shift support is limited to shifting rows by 1");
            mustBeDim = 0;
        }
        const auto& shape = x.Shape();
        size_t rank = shape.Rank();
        size_t len = shape[rank - 1];
        if (rank == 0)
            CNTK::InvalidArgument("marian::shift cannot shift scalars");
        auto remaining = CNTK::Slice(x, CNTK::Axis((int)rank-1), 0, (int)len-1);
        auto inserted = CNTK::Constant(shape.SubShape(0, rank-1), Dynamite::CurrentDataType(), 0.0, Dynamite::CurrentDevice());
        Expr res = CNTK::Splice({ inserted, remaining }, CNTK::Axis((int)rank-1));
        //x.Value()->LogToFile(L"x");
        //res.Value()->LogToFile(L"shift(x)");
        return res;
    }

    static inline Expr convert2cudnnFormat(const Expr& x)
    {
        x; // TODO: find out the semantics
        return InternalOps::NotImplemented("convert2cudnnFormat");
    }

    static inline Expr convertFromcudnnFormat(const Expr& x)
    {
        x; // TODO: find out the semantics
        return InternalOps::NotImplemented("convertFromcudnnFormat");
    }

    static inline Expr avg_pooling(const Expr& x, int height, int width, int padHeight = 0, int padWidth = 0, int strideHeight = 1, int strideWidth = 1)
    {
        x, height, width, padHeight, padWidth, strideHeight, strideWidth; // TODO: implement these in CNTK Dynamite
        return InternalOps::NotImplemented("avg_pooling");
    }

    static inline Expr max_pooling(const Expr& x, int height, int width, int padHeight = 0, int padWidth = 0, int strideHeight = 1, int strideWidth = 1)
    {
        x, height, width, padHeight, padWidth, strideHeight, strideWidth; // TODO: implement these in CNTK Dynamite
        return InternalOps::NotImplemented("max_pooling");
    }

    static inline Expr pooling_with_masking(const Expr& x, Expr mask, int width, bool isEven = false)
    {
        x, mask, width, isEven; // TODO: implement these in CNTK Dynamite
        return InternalOps::NotImplemented("pooling_with_masking");
    }

    namespace rnn // RNN has special ops that must be emulated for now
    {
        static inline Expr gruOps(const std::vector<Expr>& x, bool)
        {
            return x.front();
        }
        static inline Expr lstmOpsC(const std::vector<Expr>& x)
        {
            return x.front();
        }
        static inline Expr lstmOpsO(const std::vector<Expr>& x)
        {
            return x.front();
        }
    };

    // added for CNTK: same as graph->constant() without the graph
    static inline Expr constant(const Shape& npShape, const CNTK::ParameterInitializer& init) { return InternalOps::Constant(npShape, init, /*isVolatile=*/false); }

    // -----------------------------------------------------------------------
    // ExpressionGraph
    // -----------------------------------------------------------------------

    struct Parameters
    {
        std::map<std::string, CNTK::Parameter> m_allParametersMap;
        std::vector<CNTK::Parameter> m_allParameters;
        std::map<std::string, Expr> getMap() const
        {
            std::map<std::string, Expr> res; // Parameters does not cast to Expr, so we must make a copy here. It's just shared_ptrs.
            for (let& i : m_allParametersMap)
                res[i.first] = (Expr)i.second;
            return res;
        }
        // CNTK API:
        const std::vector<CNTK::Parameter>& get() const { return m_allParameters; }
    };

    class ExpressionGraph
    {
    public:
        ExpressionGraph() {}
        Ptr<Parameters> params() const { return m_parameters; }
        void clear() { }
        void reserveWorkspaceMB(size_t) { }
        // TODO: what is Marian's device id of the CPU?
        void setDevice(size_t device = 0)
        {
            Dynamite::SetCurrentDevice(CNTK::DeviceDescriptor::GPUDevice((unsigned int)device));
        }
        size_t getDevice() { return Dynamite::CurrentDevice().Id(); }
        struct Backend // fake Backend for getBackend()
        {
            void setDevice(size_t gpuId) { Dynamite::SetCurrentDevice(DeviceDescriptor::GPUDevice((unsigned int)gpuId)); }
        };
        Backend* getBackend() { static Backend s_backend; return &s_backend; }
        void setInference(bool inference) { m_inferenceOnly = inference; }
        Expr constant(const Shape& npShape, const CNTK::ParameterInitializer& init) const { return InternalOps::Constant(npShape, init, /*isVolatile=*/m_inferenceOnly); }
        Expr zeros(const Shape& npShape) const { return InternalOps::Constant(npShape, inits::zeros, /*isVolatile=*/m_inferenceOnly); }
        // TODO: namespace; lots more
        Expr param(const std::string& name, const Shape& shape, const CNTK::ParameterInitializer& init, bool fixed = false)
        {
            fixed; // TODO
            auto& pp = *params();
            auto viewShape = mappers::ToNDShape(shape); // convert to CNTK's column-major viewShape
            auto iter = pp.m_allParametersMap.find(name);
            if (iter == pp.m_allParametersMap.end()) // case 1: create a new parameter
            {
                if (init.Contains(L"from_vector")) // our fake 
                {
                    // BUGBUG: This keeps a reference to the vector, not a copy, which only works inside a single expression, if at all.
                    const auto& initData = init[L"from_vector"].Value<CNTK::NDArrayView>();
                    if (initData.Shape().TotalSize() != viewShape.TotalSize())
                        CNTK::InvalidArgument("marian::constant: vector size does not match viewShape");
                    // copy the supplied CPU buffer, which may be a temporary, to a GPU-side NDArrayView
                    auto initVal = initData.AsShape(viewShape)->DeepClone(Dynamite::CurrentDevice(), /*readOnly=*/false);
                    auto p = CNTK::Parameter(initVal);
                    pp.m_allParametersMap.insert(std::make_pair(name, p));
                    pp.m_allParameters.push_back(p);
                    m_allGradients[p] = nullptr;
                    return (CNTK::Variable)p;
                }
                else
                {
                    auto p = CNTK::Parameter(viewShape, CNTK::DataType::Float, init, Dynamite::CurrentDevice(), std::wstring(name.begin(), name.end())); // copy it (possibly to a different device)
                    pp.m_allParametersMap.insert(std::make_pair(name, p));
                    pp.m_allParameters.push_back(p);
                    m_allGradients[p] = nullptr;
                    return (CNTK::Variable)p;
                }
            }
            else  // case 2: retrieve an existing parameter
            {
                const auto& p = iter->second;
                if (p.Shape() != viewShape)
                    CNTK::InvalidArgument("marian::param: Requested shape for existing parameter '%s' does not match original shape", name.c_str());
                return (CNTK::Variable)p;
            }
        }
        Expr get(std::string name) const
        {
            auto& pp = *params();
            //if (!namespace_.empty())
            //    name = namespace_ + "::" + name;
            auto iter = pp.m_allParametersMap.find(name);
            if (iter != pp.m_allParametersMap.end())
                return (CNTK::Variable)iter->second;
            else
                return nullptr;
        }
        Expr dropout(float dropRate, const Shape& npShape)    { return InternalOps::DropoutMask(dropRate, mappers::ToNDShape(npShape)); }
        Expr dropout(float dropRate, const ShapeProxy& shape) { return InternalOps::DropoutMask(dropRate, shape.GetNDShape()); }
        // forward/backward
        void forward() { }
        void forwardNext() { }
        // BREAKING CHANGE: must pass the root for backprop
        void backward(const Expr& root) { backprop(root); }
        void backprop(const Expr& root)
        {
            root.Backward(m_allGradients);
        }
        // methods presently not supported
        void load(const std::string&, bool) { NOT_IMPLEMENTED; }
        void save(const std::string&) { NOT_IMPLEMENTED; }
        bool fits() { NOT_IMPLEMENTED; }
        void setReloaded(bool) { NOT_IMPLEMENTED; }
    private:
        Ptr<Parameters> m_parameters = make_shared<Parameters>();
        bool m_inferenceOnly = false;
        friend class OptimizerWrapper;
        std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr> m_allGradients;
    };

    enum AlgorithmType { Sgd, Adam };
    class OptimizerWrapper
    {
    public:
        // TODO: more parameters; schedules, lots more
        OptimizerWrapper(float eta, AlgorithmType algorithmType)
        {
            switch (algorithmType)
            {
            case AlgorithmType::Sgd:
                m_LazyCreateLearner = [=](const Ptr<ExpressionGraph>& graph)
                {
                    return CNTK::SGDLearner(graph->params()->get(),
                                            CNTK::LearningRateSchedule(std::vector<double>{ eta }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1)/*,
                                            AdditionalLearningOptions additionalOptions = AdditionalLearningOptions()*/);
                };
                break;
            case AlgorithmType::Adam:
                m_LazyCreateLearner = [=](const Ptr<ExpressionGraph>& graph)
                {
                    return CNTK::AdamLearner(graph->params()->get(),
                                            CNTK::LearningRateSchedule(std::vector<double>{ eta }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            CNTK::MomentumSchedule(std::vector<double>{ 0.9 }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            /*unitGain=*/true,
                                            CNTK::MomentumSchedule(std::vector<double>{ 0.999 }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            /*epsilon=*/1e-8, /*adamax=*/false/*
                                            AdditionalLearningOptions additionalOptions = AdditionalLearningOptions()*/);
                };
                break;
            }
        }
        // TODO: sample count?
        void update(const Ptr<ExpressionGraph>& graph)
        {
            if (!m_learner)
                m_learner = m_LazyCreateLearner(graph);
            // sample count 1 disables all rescaling, and expects the gradient as the user wants it
            m_learner->Update(graph->m_allGradients, /*trainingSampleCount=*/1);
        }
    private:
        std::function<CNTK::LearnerPtr(const Ptr<ExpressionGraph>&)> m_LazyCreateLearner;
        CNTK::LearnerPtr m_learner;
    };
    template<AlgorithmType algorithmType>
    Ptr<OptimizerWrapper> Optimizer(float eta)
    {
        return New<OptimizerWrapper>(eta, algorithmType);
    }
} // namespace marian

namespace cnpy // needed for Amun model's load/save functions (currently not supported)
{
    static inline std::map<string, marian::inits::Internal::NumpyObject> npz_load(const string&) { NOT_IMPLEMENTED; }
    static inline void npz_save(const string&, const string&, const float*, unsigned *, unsigned, const std::string&) { NOT_IMPLEMENTED; }
} // namespace cnpy

#endif // __MARIAN
