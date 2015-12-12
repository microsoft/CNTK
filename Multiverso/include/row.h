// WARNING: Deprecated code, not used this time, may be refined in the future.

#ifndef _MULTIVERSO_ROW_H_
#define _MULTIVERSO_ROW_H_

namespace multiverso
{
    class RowBase
    {
    public:
        virtual int Add(void *delta) = 0;
        virtual void *Get() = 0;
        virtual size_t GetMemSize() = 0;
    };

    template <typename T>
    class Row : public RowBase
    {
    public:
        Row(size_t col_count, void *pt);
        ~Row();
        int Add(void *delta) override;
        void *Get() override { return data_; }
        size_t GetMemSize() override { return col_count_ * sizeof(T); }

    protected:
        T *data_;
        size_t col_count_;
    };

    class RowCreator
    {
    public:
        static RowBase *CreateRow(int table_id, void *pt);
    };
}

#endif // _MULTIVERSO_ROW_H_ 