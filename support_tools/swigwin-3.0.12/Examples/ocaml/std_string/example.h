/* File : example.h -- stolen from the guile std_vector example */

#include <string>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdlib.h>
#include <locale.h>

std::string from_wstring_with_locale( const std::wstring source,
				      const std::string locale ) {
    const char *current_locale = setlocale( LC_CTYPE, locale.c_str() );
    int required_chars = wcstombs( NULL, source.c_str(), 0 );
    std::string s;
    char *temp_chars = new char[required_chars + 1];
    temp_chars[0] = 0;
    wcstombs( temp_chars, source.c_str(), required_chars + 1 );
    s = temp_chars;
    delete [] temp_chars;
    setlocale( LC_CTYPE, current_locale );
    return s;
}

std::wstring to_wstring_with_locale( const std::string source,
				     const std::string locale ) {
    const char *current_locale = setlocale( LC_CTYPE, locale.c_str() );
    int required_chars = mbstowcs( NULL, source.c_str(), 0 );
    std::wstring s;
    wchar_t *temp_chars = new wchar_t[required_chars + 1];
    temp_chars[0] = 0;
    mbstowcs( temp_chars, source.c_str(), required_chars + 1 );
    s = temp_chars;
    delete [] temp_chars;
    setlocale( LC_CTYPE, current_locale );
    return s;
}
