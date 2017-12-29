%module bloody_hell

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) kMaxIOCTLSpaceParmsSize;

#ifdef SWIGSCILAB
%rename(Parms) sm_channel_ix_dump_parms;
#endif

#define kMaxIOCTLSpaceParmsSize 128

%{
#define kMaxIOCTLSpaceParmsSize 128
  %}

%inline %{
typedef struct sm_channel_ix_dump_parms {
     unsigned        data[(kMaxIOCTLSpaceParmsSize - ((4*sizeof(int)) + (2*sizeof(unsigned))))/sizeof(unsigned)];
} SM_CHANNEL_IX_DUMP_PARMS;
 %}



