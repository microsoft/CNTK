// test of std::queue
%module li_std_queue

%include std_queue.i


%template( IntQueue ) std::queue< int >;
