%include <perlstrings.swg>

%fragment("<XMLCh.h>","header") 
%{
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/util/XMLUTF8Transcoder.hpp>
%}

%fragment("SWIG_UTF8Transcoder","header",fragment="<XMLCh.h>") {
SWIGINTERN XERCES_CPP_NAMESPACE::XMLTranscoder* 
SWIG_UTF8Transcoder() {
  using namespace XERCES_CPP_NAMESPACE;
  static int init = 0;
  static XMLTranscoder* UTF8_TRANSCODER  = NULL;
  static XMLCh* UTF8_ENCODING = NULL;  
  if (!init) {
    XMLTransService::Codes failReason;
    XMLPlatformUtils::Initialize(); // first we must create the transservice
    UTF8_ENCODING = XMLString::transcode("UTF-8");
    UTF8_TRANSCODER = XMLPlatformUtils::fgTransService->makeNewTranscoderFor(UTF8_ENCODING,
									     failReason,
									     1024);
    init = 1;
  }
  return UTF8_TRANSCODER;
}
}
   
%fragment("SWIG_AsXMLChPtrAndSize","header",fragment="SWIG_AsCharPtrAndSize",fragment="SWIG_UTF8Transcoder") {
SWIGINTERN int
SWIG_AsXMLChPtrAndSize(SV *obj, XMLCh **val, size_t* psize, int *alloc)
{
  if (!val) {
    return SWIG_AsCharPtrAndSize(obj, 0,  0, 0);
  } else {
    size_t size;
    char *cptr = 0;
    int calloc = SWIG_OLDOBJ;
    int res = SWIG_AsCharPtrAndSize(obj, &cptr, &size, &calloc);
    if (SWIG_IsOK(res)) {
      STRLEN length = size - 1;
      if (SvUTF8(obj)) {
	unsigned int charsEaten = 0;
	unsigned char* sizes = %new_array(size, unsigned char);
	*val = %new_array(size, XMLCh);
	unsigned int chars_stored = 
	  SWIG_UTF8Transcoder()->transcodeFrom((const XMLByte*) cptr,
					       (unsigned int) length,
					       (XMLCh*) *val, 
					       (unsigned int) length,
					       charsEaten,
					       (unsigned char*)sizes
					       );
	%delete_array(sizes);
	// indicate the end of the string
	(*val)[chars_stored] = '\0';
      } else {
	*val = XERCES_CPP_NAMESPACE::XMLString::transcode(cptr);
      }
      if (psize) *psize = size;
      if (alloc) *alloc = SWIG_NEWOBJ;
      if (calloc == SWIG_NEWOBJ) %delete_array(cptr);
      return SWIG_NEWOBJ;    
    } else {
      return res;
    }
  }
}
}

%fragment("SWIG_FromXMLChPtrAndSize","header",fragment="SWIG_UTF8Transcoder") {
SWIGINTERNINLINE SV *
SWIG_FromXMLChPtrAndSize(const XMLCh* input, size_t size)
{
  SV *output = sv_newmortal();
  unsigned int charsEaten = 0;
  int length  = size;                                        // string length
  XMLByte* res = %new_array(length * UTF8_MAXLEN, XMLByte);          // output string
  unsigned int total_chars =
    SWIG_UTF8Transcoder()->transcodeTo((const XMLCh*) input, 
				       (unsigned int) length,
				       (XMLByte*) res,
				       (unsigned int) length*UTF8_MAXLEN,
				       charsEaten,
				       XERCES_CPP_NAMESPACE::XMLTranscoder::UnRep_Throw
				       );
  res[total_chars] = '\0';
  sv_setpv((SV*)output, (char *)res );
  SvUTF8_on((SV*)output);
  %delete_array(res);
  return output;
}
}

%fragment("SWIG_XMLStringNLen","header") {
size_t
SWIG_XMLStringNLen(const XMLCh* s, size_t maxlen)
{
  const XMLCh *p;
  for (p = s; maxlen-- && *p; p++)
    ;
  return p - s;
}
}

%init {
  if (!SWIG_UTF8Transcoder()) {
    croak("ERROR: XML::Xerces: INIT: Could not create UTF-8 transcoder");
  }
}


%include <typemaps/strings.swg>
%typemaps_string(%checkcode(UNISTRING), %checkcode(UNICHAR),
		 XMLCh, XMLCh,
		 SWIG_AsXMLChPtrAndSize, 
		 SWIG_FromXMLChPtrAndSize,
		 XERCES_CPP_NAMESPACE::XMLString::stringLen,
		 SWIG_XMLStringNLen,
		 "<XMLCh.h>", INT_MIN, INT_MAX);


