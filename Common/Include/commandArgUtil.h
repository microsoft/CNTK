//
// <copyright file="commandArgUtil.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include "Basics.h"
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <stdint.h>

using namespace std;

    // helper for numeric parameter arguments for multiple layers
    // This parses arguments of the form a:b*n:c where b gets duplicated n times and c unlimited times.
namespace Microsoft { namespace MSR { namespace CNTK {

#define FUNCTIONOPEN "("
#define OPENBRACES "[{(\""
#define CLOSINGBRACES "]})\""

    static const std::string::size_type npos = (std::string::size_type )-1;

    // These are the constants associated with the "ResolveVariables" method.
    static const std::string openBraceVar = "$";
    static const std::string closingBraceVar = "$";
    static const std::string forbiddenCharactersInVarName = ",/<>?;':\"[]{}\\|!@#%^&*()+=~` \t\n"; 
    static const std::string forbiddenCharactersInVarNameEscapeWhitespace = ",/<>?;':\"[]{}\\|!@#%^&*()+=~` \\t\\n"; 
    static const std::size_t openBraceVarSize = openBraceVar.size();
    static const std::size_t closingBraceVarSize = openBraceVar.size();

    // Trim - trim white space off the start and end of the string
    // str - string to trim
    // NOTE: if the entire string is empty, then the string will be set to an empty string
    void Trim(std::string& str);
    class ConfigValue;
    typedef std::map<std::string, ConfigValue, nocase_compare> ConfigDictionary;
    class ConfigParameters;
    std::string::size_type ParseKeyValue(const std::string& token, std::string::size_type pos, ConfigParameters& dict);

    // ConfigValue - value of one configuration parameter
    // Parses from string to resultant value on assignment
    class ConfigValue : public std::string
    {
        std::string m_configName;       // name of this configuration, e.g. for error messages, optional
        const ConfigParameters* m_parent;     // keep track of parent pointer
    public:
        std::string Name() const 
        {return m_configName;}
        const ConfigParameters* Parent() const
        {return m_parent;}
        void SetParent(const ConfigParameters* parent)
        {m_parent = parent;}
    protected:
        // Constructor with a parent pointer. NOTE: this MUST be used with care. Parent lifetime must be longer than ConfigValue lifetime
        ConfigValue (const std::string & val, const std::string & name, const ConfigParameters* parent) : std::string (val)
        { 
            m_configName = name;
            m_parent = parent;
        }
        // only allow these classes to construct ConfigValues with parent pointers, they are meant as intermediate values only
        friend class ConfigParameters;
        friend class ConfigArray;
    public:
        ConfigValue (const std::string & val, const std::string & name) : std::string (val), m_configName(name), m_parent(NULL)
        {
        }
        ConfigValue (const std::string & val) : std::string (val), m_parent(NULL) 
        { }
        // empty constructor so ConfigValue can be contained in a std::map (requires default constructor)
        ConfigValue () : m_parent(NULL)
        { }
        // it auto-casts to the common types
        // Note: This is meant to read out a parameter once to assign it, instead of over again.
        operator std::string () const { return *this; } // TODO: does not seem to work
        operator const char * () const { return c_str(); }
        operator std::wstring () const { return msra::strfun::utf16(*this); } 
        operator double () const
        {
            char * ep;          // will be set to point to first character that failed parsing
            double value = strtod (c_str(), &ep);
            if (empty() || *ep != 0)
            {   // check for infinity since strtod() can't handle it
                if (*ep && _strnicmp("#inf",ep, 4) == 0)
                    return std::numeric_limits<double>::infinity();
                RuntimeError ("ConfigValue (double): invalid input string");
            }
            return value;
        }
        operator float () const { return (float) (double) *this; }
    private:
        long tolong() const
        {
            char * ep;          // will be set to point to first character that failed parsing
            long value = strtol(c_str(), &ep, 10);
            if (empty() || *ep != 0)
                RuntimeError("ConfigValue (long): invalid input string");
            return value;
        }
        unsigned long toulong() const
        {
            char * ep;          // will be set to point to first character that failed parsing
            unsigned long value = strtoul(c_str(), &ep, 10);
            if (empty() || *ep != 0)
                RuntimeError("ConfigValue (unsigned long): invalid input string");
            return value;
        }
    public:
        operator short () const
        {
            long val = tolong();
            short ival = (short) val;
            if (val != ival)
                RuntimeError ("ConfigValue (short): integer argument expected");
            return ival;
        }
        operator unsigned short () const
        {
            unsigned long val = toulong();
            unsigned short ival = (unsigned short) val;
            if (val != ival)
                RuntimeError ("ConfigValue (unsigned short): integer argument expected");
            return ival;
        }
        operator int () const
        {
            long val = tolong();
            int ival = (int) val;
            if (val != ival)
                RuntimeError ("ConfigValue (int): integer argument expected");
            return ival;
        }
        operator unsigned int () const
        {
            unsigned long val = toulong();
            unsigned int ival = (unsigned int) val;
            if (val != ival)
                RuntimeError ("ConfigValue (unsigned int): integer argument expected");
            return ival;
        }
//#if (SIZE_MAX != ULONG_MAX)     // on x64 GCC unsigned long == size_t, i.e. we'd get an ambigous declaration
#ifdef _MSC_VER // somehow the above check does not work on GCC/Cygwin, causing an ambiguous declaration
        operator unsigned long() const { return toulong(); }
        operator long() const { return tolong(); }
#endif
        operator int64_t () const
        {
            char * ep;          // will be set to point to first character that failed parsing
            int64_t value = _strtoi64 (c_str(), &ep, 10);
            if (empty() || *ep != 0)
                RuntimeError ("ConfigValue (int64_t): invalid input string");
            return value;
        }
        operator uint64_t () const
        {
            char * ep;          // will be set to point to first character that failed parsing
            uint64_t value = _strtoui64 (c_str(), &ep, 10);
            if (empty() || *ep != 0)
                RuntimeError ("ConfigValue (uint64_t): invalid input string");
            return value;
        }
        // size_t is the same as uint64_t()
        //operator size_t () const
        //{
        //    uint64_t val = (uint64_t) *this;
        //    size_t ival = (size_t) val;
        //    if (val != ival)
        //        RuntimeError ("ConfigValue (size_t): integer argument expected");
        //    return ival;
        //}
        operator bool () const
        {
            const auto & us = *this;
            if (us == "t" || us == "true" || us == "T" || us == "True" || us == "TRUE" || us == "1")
                return true;
            if (us == "f" || us == "false" || us == "F" || us == "False" || us == "FALSE" || us == "0" || us == "")
                return false;
            RuntimeError ("ConfigValue (bool): boolean argument expected");
            // TODO: do we want to allow accept non-empty strings and non-0 numerical values as 'true'?
        }

        //ReplaceAppend - replace an existing value with another value, or append if it appears to be a "set" type
        ConfigValue& ReplaceAppend(const std::string& configValue)
        {
            static const std::string openBraces = "[";

            // if we have openBraces, append (it's a group)
            if (length() > 0 && openBraces.find(configValue[0]) != npos)
            {
                // append another value to the current value, add a space separator
                append(" ");
                append(configValue);
            }
            else  // otherwise replace
            {
                this->assign(configValue);
            }
            return *this;
        }
    };

    // parse config parameters on separators, and keep track of configuration names
    class ConfigParser
    {
    protected:
        char m_separator;
        mutable std::string m_configName;       // name of this configuration, e.g. for error messages, optional

        // parse at the file level, can be overridden for "section of file" behavior
        virtual void FileParse(const std::string& stringParse)
        {Parse(stringParse);}

    public:
        ConfigParser (char separator, const std::string & configname) : m_separator(separator), m_configName (configname) 
        { }
        ConfigParser (char separator, const std::wstring & configname) : m_separator(separator)
        {m_configName = msra::strfun::utf8(configname); }

        ConfigParser(char separator) : m_separator(separator), m_configName("unknown") {}
        ConfigParser(const ConfigParser& configParser)
        {
            m_separator = configParser.m_separator;
            m_configName = configParser.m_configName;
        }
        ConfigParser(const ConfigParser&& configParser)
        {
            m_separator = configParser.m_separator;
            m_configName = move(configParser.m_configName);
        }
        ConfigParser& operator=(const ConfigParser& configParser) = default;

    public:
        // FindBraces - find matching braces in a string starting at the current position
        // str - string to search
        // tokenStart - start location in the string to search
        // returns: position of last brace, -1 if no braces at current position
        static std::string::size_type FindBraces(const std::string& str, std::string::size_type tokenStart)
        {
            static const std::string openBraces = OPENBRACES; // open braces and quote
            static const std::string closingBraces = CLOSINGBRACES; // close braces and quote
            auto braceFound = openBraces.find(str[tokenStart]);
            auto len = str.length();
            if (braceFound == npos || tokenStart >= len)
                return npos;

            std::vector<std::string::size_type> bracesFound;
            std::string::size_type current, opening;
            current = opening = tokenStart; //str.find_first_of(openBraces, tokenStart);

            // create a brace pair for string searches
            std::string braces;
            braces += openBraces[braceFound];
            braces += closingBraces[braceFound];

            // search for end brace or other nested layers of this brace type
            while (current != npos && current+1 < len)
            {
                current = str.find_first_of(braces, current+1);
                // check for a nested opening brace
                if (current == npos)
                    break;

                // found a closing brace
                if (str[current] == braces[1])
                {
                    // no braces on the stack, we are done
                    if (bracesFound.empty())
                    {
                        return current;
                    }
                    // have braces on the stack, pop the current one off
                    opening = bracesFound.back();
                    bracesFound.pop_back();
                }
                else  
                // found another opening brace, push it on the stack
                {
                    bracesFound.push_back(opening);
                    opening = current;
                }
            }
            // if we found unmatched parenthesis, throw an exception
            if (opening != npos)
                RuntimeError("unmatched bracket found in parameters");
            return current;
        }

        // Parse - Parse the string; segment string by top-level a=b expressions and call (virtual) ParseValue() on them.
        // This function is used at lots of places for various purposes.
        //  - (ConfigParameters from file) config-file parsing passes in expressions of the type a1=b1 \n a2=b2 \n ..., creates a ConfigDictionary entry for each top-level a=b expression, where b can be a block in braces
        //  - (ConfigParameters) right-hand side that is an array of parameters [ a1=b1; a2=b2 ...], with surrounding braces
        //  - (ConfigValue) individual values are also parsed
        //  - (ConfigArray) same as ConfigValue--the array syntax (':') is not parsed here
        //    The above all allow ';' or newline as a separator
        //  - (NDLScript)
        //  - more to be added
        // stringParse - string to parse
        // pos - postion to start parsing at
        void Parse(const std::string& stringParse, std::string::size_type pos=0)
        {
            // list of possible custom separators
            const std::string customSeperators = "`~!@$%^&*_-+|:;,?.";
            std::string seps = ",\r\n";   // default separators
            // add braces and current separator to the separators list so we skip them
            seps += m_separator;
            std::string sepsBraces(seps);
            sepsBraces += OPENBRACES;

            // Establish string and get the first token:
            auto tokenEnd = pos;
            auto totalLength = stringParse.length();
            auto braceEnd = totalLength;
            bool contentLevel = false;  // content level (not surrounding braces)

            do
            {
                auto tokenStart = stringParse.find_first_not_of(seps, tokenEnd);
                if (tokenStart==npos)   // no more tokens
                    break;
                // skip any leading spaces
                tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
                if (tokenStart == npos)
                    break;

                auto braceEndFound = FindBraces(stringParse, tokenStart);
                bool quoteFound = false;

                if (braceEndFound != npos && tokenStart+1 < totalLength)
                {
                    if (!contentLevel)
                    {
                        tokenStart++; // skip the brace
                        // check for custom separator character
                        if (customSeperators.find(stringParse[tokenStart]) != npos)
                        {
                            char separator = stringParse[tokenStart];
                            seps[seps.length()-1] = separator;
                            sepsBraces = seps + OPENBRACES;
                            tokenStart++; // skip the separator
                        }
                        braceEnd = braceEndFound;
                        tokenEnd = tokenStart;
                        contentLevel = true;    // now at content level
                        continue;
                    }
                }

                // content level braces, just find the end of the braces
                if (braceEndFound != npos)
                {
                    if (stringParse[braceEndFound] == '"')
                    {   // for quoted string we skip the quotes
                        tokenStart++;
                        tokenEnd = braceEndFound;
                        quoteFound = true;
                    }
                    else
                    {
                        tokenEnd = braceEndFound+1; // tokenEnd is one past the character we want
                    }
                }
                else
                {
                    // find the end of the token
                    tokenEnd = stringParse.find_first_of(sepsBraces, tokenStart);

                    // now look for contained braces before the next break
                    if (tokenEnd != npos)
                        braceEndFound = FindBraces(stringParse, tokenEnd);
                    // found an embedded brace, go to matching end brace to end token
                    if (braceEndFound != npos)
                    {
                        tokenEnd = braceEndFound+1; // token includes the closing brace
                    }


                    if (tokenEnd==npos || tokenEnd > braceEnd)   // no more seperators
                    {   // use the length of the string as the boundary
                        tokenEnd = braceEnd;
                        if (tokenStart >= totalLength) // if nothing left, we are done
                            break;
                    }
                }

                // now parse the value
                if (tokenEnd > tokenStart)
                {
                    tokenEnd = ParseValue(stringParse, tokenStart, tokenEnd);
                }
                // if we hit the end of a brace block, move past the ending brace and reset
                if (tokenEnd == braceEnd)
                {
                    tokenEnd++;
                    braceEnd = totalLength;
                    seps[seps.length()-1] = m_separator;    // restore default separator
                    sepsBraces = seps + OPENBRACES;
                    contentLevel = false;
                }
                if (quoteFound)
                {   // skip the closing quote
                    tokenEnd++; 
                }
                // While we have tokens to parse
            }
            while (tokenEnd != npos);
        }

        // StripComments - This method removes the section of a config line corresponding to a comment.
        // configLine - The line within a config file to pre-process.
        // returns:
        //      If the entire line is whitespace, or if the entire line is a comment, simply return an empty string.
        //      If there is no comment, simply return the original 'configString'
        //      If there is a comment, remove the part of 'configString' corresponding to the comment
        //      Note that midline comments need to be preceded by whitespace, otherwise they are not treated as comments.
        std::string StripComments(const std::string &configLine) const
        {
            std::string::size_type pos = configLine.find_first_not_of(" \t");

            // entire line is whitespace, or it is a full line comment.
            if (pos == std::string::npos || configLine[pos] == '#')
                return "";

            // search for a comment mid line
            std::string::size_type midLineCommentPos = configLine.find_first_of('#',pos);

            // if there is no comment, simply return original string
            if(midLineCommentPos == std::string::npos)
                return configLine;

            // if we have a mid-line comment, make sure it's preceded by a whitespace character
            // otherwise, don't treat this midline comment as a comment.
            char chPrev = configLine[midLineCommentPos-1]; // this should be safe because midLineCommentPos is guaranteed to be > 0            
            return (chPrev == ' ' || chPrev == '\t') ? configLine.substr(pos, midLineCommentPos - pos) : configLine;
        }

        virtual std::string::size_type ParseValue(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd) = 0;
        std::string ReadConfigFile(const std::string &filePath);
        std::string ReadConfigFile(const std::wstring &filePath);
        std::string ReadConfigFiles(const std::string &filePaths);
        std::string ReadConfigFiles(const std::wstring &filePaths);
        std::string ResolveIncludeStatements(const std::string &configString, std::vector<std::string> &resolvedConfigFiles);
        void LoadConfigFile(const std::wstring &  filePath);
        void LoadConfigFileAndResolveVariables(const std::wstring &filePath, const ConfigParameters& config);
        void LoadConfigFiles(const std::wstring &  filePaths, const std::string *configStringToAppend=nullptr);
        void SetName(const std::wstring & name) 
        {m_configName = msra::strfun::utf8(name);}
        void SetName(const std::string & name) 
        {m_configName = name;}
        std::string Name() const 
        {return m_configName;}
    };

    // dictionary of parameters
    // care should be used when using this class it has parent links to stack variables which are assumed to exist and have lifetimes that are allocated and freed in a FIFO manner.
    // If this is not the case for a particular variable (stored in a class or something), you must call ClearParent() to disconnect it from it's parents before they are freed.
    // usage: This class is intended to be used as local variables where the "parent" parameters have lifetimes longer than the "child" parameters
    // for example:
    // int wmain(int argc, wchar_t* argv[]) {
    //    ConfigParameters config = ConfigParameters::ParseCommandLine(argc, argv);
    //    A(config);
    // }
    // void A(const ConfigParameters& config) {ConfigParameters subkey1 = config("a"); /* use the config params */ B(subkey);}
    // void B(const ConfigParameters& config) {ConfigParameters subkey2 = config("b"); /* use the config params */}
    class ConfigParameters : public ConfigParser, public ConfigDictionary
    {
        // WARNING: the parent pointer use requires parent lifetimes be longer than or equal to children.
        const ConfigParameters* m_parent;
    public:
        // empty constructor 
        ConfigParameters () : ConfigParser(';'), m_parent(NULL) { } // parameterless version for subConfig Dictionaries

        // construct ConfigParameters from a ConfigValue, propogate the config Name, and parent pointer
        ConfigParameters(const ConfigValue& configValue) : ConfigParser(';',configValue.Name()), m_parent(configValue.Parent())
        {
            std::string configString = configValue;
            Parse(configString);
        }

    //private:
        // copy and move constructors
        ConfigParameters(const ConfigParameters& configValue) : ConfigParser(configValue)
        {
            *this = configValue;
        }
        ConfigParameters(const ConfigParameters&& configValue) : ConfigParser(move(configValue))
        {
            *this = move(configValue);
        }
        ConfigParameters& operator=(const ConfigParameters& configValue)
        {
            this->ConfigParser::operator=(configValue);
            this->ConfigDictionary::operator=(configValue); 
            this->m_parent = configValue.m_parent;
            return *this;
        }
        ConfigParameters& operator=(const ConfigParameters&& configValue)
        {
            this->ConfigParser::operator=(configValue);
            this->ConfigDictionary::operator=(configValue); 
            this->m_parent = configValue.m_parent;
            return *this;
        }
        // hide new so only stack allocated
        void * operator new(size_t /*size*/) {}
    public:
        // explicit copy function. Only to be used when a copy must be made.
        // this also clears out the parent pointer, so only local configs can be used
        ConfigParameters& CopyTo(ConfigParameters& copyTo) const
        {
            copyTo = *this;
            copyTo.ClearParent();
            return copyTo;
        }

        // clear the parent link, important when storing ConfigParameters in a class where parent lifetime is not guaranteed
        void ClearParent()
        {
            m_parent = NULL;
        }

        const ConfigParameters* GetParent() const
        {
            return m_parent;
        }

        // parse a 'key=value' pair and insert in the ConfigDictionary
        std::string::size_type ParseValue(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd)
        {
            // skip leading spaces
            tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
            auto keyEnd = stringParse.find_first_of("=" OPENBRACES, tokenStart);
            std::string value;

            // if no value is specified, it's a boolean variable and set to true
            if (keyEnd == npos || keyEnd >= tokenEnd)
            {
                auto key = stringParse.substr(tokenStart, tokenEnd-tokenStart);
                Trim(key);
                value = "true";
                if (!key.empty())
                    Insert(key, value);
                return tokenEnd;
            }

            // get the key
            auto key = stringParse.substr(tokenStart, keyEnd-tokenStart);
            Trim(key);
            tokenStart = keyEnd;
            if (stringParse[keyEnd] == '=')
                ++tokenStart;

            // skip any spaces before the second token
            tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
            std::string::size_type substrSize = tokenEnd - tokenStart;

            auto bracesEnd = FindBraces(stringParse, tokenStart);

            // if braces found, we modify the token end according to braces
            if (bracesEnd != npos)
            {   // include the trailing brace
                tokenEnd = bracesEnd+1;
                substrSize = tokenEnd - tokenStart;

                // for quote delimited string remove quotes
                if (stringParse[tokenStart] == '"')
                {
                    tokenStart++;
                    substrSize -= 2;    // take out the quotes
                }
            }

            if (substrSize == 0)
                return npos;

            // get the value
            value = stringParse.substr(tokenStart, substrSize);
            Trim(value);
            // add the value to the dictionary if both values are valid
            if (!key.empty() && !value.empty())
                Insert(key, value);
            return tokenEnd;
        }

         // Insert - insert a new name and value into the dictionary
        void Insert (const std::wstring & name, const std::string & val)
        {
            Insert(msra::strfun::utf8(name), val);
        }

        // Insert - insert a new name and value into the dictionary
        void Insert (const std::string & name, const std::string & val)
        {
            auto iter = find(name);
            if (iter != end())
            {
                // replace or append the value
                iter->second.ReplaceAppend(val);
            }
            else
            {
                std::string fullName = m_configName +  ":" + name;
                auto res = ConfigDictionary::insert (std::make_pair (name, ConfigValue(val, fullName, this)));
                if (!res.second)    // no insertion was made
                    RuntimeError ("configparameters: duplicate parameter definition for %s", fullName.c_str());
            }
        }

        // Insert - insert an 'name=value' string into the dictionary
        void Insert (const std::string &str)
        {
            ParseValue(str, 0, str.length());
        }

        bool Exists(const std::wstring & name) const {return Exists(msra::strfun::utf8(name));}
        bool Exists(const std::string & name) const 
        {
            if (find(name) != end())
                return true;
            // now check parent if we have one
            if (m_parent != NULL)
                return m_parent->Exists(name);
            return false;
        }

        // ExistsCurrent - check to see if a key exists in THIS config, don't check parent
        bool ExistsCurrent(const std::string & name) const 
        {
            return  (find(name) != end());
        }

        // dict(name, default) for strings
        ConfigValue operator() (const std::wstring & name, const wchar_t *defaultvalue) const
        {
            return operator()(msra::strfun::utf8(name), defaultvalue);
        }

        // dict(name, default) for strings
        ConfigValue operator() (const std::string & name, const wchar_t *defaultvalue) const
        {
            return operator()(name, msra::strfun::utf8(defaultvalue).c_str());
        }
        // dict(name, default) for strings
        ConfigValue operator() (const std::wstring & name, const char *defaultvalue) const
        {
            return operator()(msra::strfun::utf8(name), defaultvalue);
        }
        // dict(name, default) for strings
        ConfigValue operator() (const std::string & name, const char *defaultvalue) const
        {
            ConfigValue value = Find(name, defaultvalue);
            return value;
        }
        ConfigValue Find(const std::string & name, const char *defaultvalue=NULL) const
        {
            auto iter = find(name);
            ConfigValue result;
            // if we aren't found, or they want the default value
            if (iter == end() || iter->second == "default")
            {
                // not found but the parent exists, check there
                if (iter == end() && m_parent != NULL)
                {
                    result = m_parent->Find(name, defaultvalue);
                }
                // no parent, so use default value
                else if (defaultvalue != NULL)
                {
                    std::string fullName = m_configName+":"+name;
                    result = ConfigValue(defaultvalue,fullName, this);
                }
            }
            else
            {
                std::string rhs = iter->second;
                rhs = this->ResolveVariables(rhs);
                std::string fullName = m_configName + ":" + name;
                result = ConfigValue(rhs, fullName, this);
            }
            return result;
        }

        // ResolveVariablesInSingleLine - In this method we replace all substrings of 'configLine' of the form "$varName$"
        //     (where varName is a variable name), with the value of the "varName" variable in config.
        //     We search up the config tree for the value, and we throw an error if we don't find it.
        //     Note that this process is recursive.  Take the following example: A=1; B=$A$; C=$B$.
        //     In this example, calling ResolveVariables with $B$, would see B=$A$, then look up the value 
        //     of A and see A=1, and it would then replace the string "$B$" with the string "1".
        //     Note that this method ignores comments in 'configString' (though they should probably already be 
        //     removed from 'configString' before calling this method), and it doesn't allow 'varName' to include 
        //     any whitespace characters.  If an opening "$" is found without a closing "$", an exception is thrown.
        // configString - the string that you would like to resolve variables in.
        // returns: A copy of 'configString' with all the variables resolved.
        std::string ResolveVariablesInSingleLine(const std::string &configLine) const
        {
            // ensure that this method was called on a single line (eg, no newline characters exist in 'configLine').
            if (configLine.find_first_of("\n") != std::string::npos)
                throw std::logic_error ("\"ResolveVariablesInSingleLine\" shouldn't be called with a string containing a newline character");

            std::string newConfigLine = StripComments(configLine);
            std::size_t start = newConfigLine.find_first_of(openBraceVar);
            std::size_t end = 0;
            while (start != std::string::npos)
            {
                // search for whitespace or closing brace.
                end = newConfigLine.find_first_of(closingBraceVar + forbiddenCharactersInVarName, start + openBraceVarSize);

                // ensure that a closing brace exists for every opening brace.  
                // Also ensure that there is no whitespace between the opening and closing braces.
                if (end == std::string::npos)
                    RuntimeError ("\"%s\" found without corresponding closing \"%s\": %s:%s", openBraceVar.c_str(), closingBraceVar.c_str(), m_configName.c_str(), newConfigLine.c_str());

                if(newConfigLine[end] != '$')
                    RuntimeError ("Forbidden characters found between \"%s\" and \"%s\".  Variable names cannot any of the following characters: %s. %s:%s",
                                  openBraceVar.c_str(), closingBraceVar.c_str(), forbiddenCharactersInVarNameEscapeWhitespace.c_str(), m_configName.c_str(), newConfigLine.c_str());

                // end + 1 - start = the length of the string, including opening and closing braces.
                std::size_t varLength = (end + 1 - start) - (openBraceVarSize + closingBraceVarSize);
                std::string varName = newConfigLine.substr(start + openBraceVarSize, varLength);

                // Note that this call to "Find" can trigger further substitutions of the form $varName2$ -> varValue2,
                // thus making this search process recursive.
                std::string varValue = this->Find(varName);

                if (varValue.empty())
                    RuntimeError ("No variable found with the name %s.  Parsing of string failed: %s:%s",
                                  varName.c_str(), m_configName.c_str(), newConfigLine.c_str());
                
                if (varValue.find_first_of("\n") != std::string::npos)
                    throw std::logic_error ("Newline character cannot be contained in the value of a variable which is resolved using $varName$ feature");


                // Replace $varName$ with 'varValue'.  Then continue the search for other variables in 'newConfigLine' string,
                // starting at the point in the 'newConfigLine' string right after 'varValue' (all variables prior to this point
                // have already been resolved, due to recursion)
                newConfigLine.replace(start, varLength + openBraceVarSize + closingBraceVarSize, varValue);
                start = newConfigLine.find_first_of(openBraceVar, start + varValue.size());
            }

            return newConfigLine;
        }


        // ResolveVariables - In this method we replace all instances of substrings of 'configString' of the form "$varName$"
        //     (where varName is a variable name), with the value of the "varName" variable in config.  We do this by calling 
        //     the 'ResolveVariablesInSingleLine' function on every line of 'configString'. See 'ResolveVariablesInSingleLine' method
        //     for more details.  Note that if there are no newlines in 'configString', then we don't append any newlines to it.
        //     This is important, because when this function is called recursively (eg, from inside the "Find" method, in order to
        //     to resolve something like "$A$" in a string like "$A$\$B$"), we shouldn't insert newlines where they didn't already exist.
        // configString - the string that you would like to resolve variables in.
        // returns: A copy of 'configString' with all the variables resolved.
        std::string ResolveVariables(const std::string &configString) const
        {
            std::string newConfigString;
            if (configString.find_first_of("\n") != std::string::npos)
            {
                // if 'configString' contains newlines, put them back after resolving each line.
                std::vector<std::string> configLines = msra::strfun::split(configString, "\n");
                for (auto configLine : configLines)
                {
                    newConfigString += ResolveVariablesInSingleLine(configLine) + "\n";
                }
            }
            else 
            {
                // if 'configString' doesn't contain any newlines, don't append a newline.
                newConfigString = ResolveVariablesInSingleLine(configString);
            }

            return newConfigString;
        }

        // dict(name): read out a mandatory parameter value
        ConfigValue operator() (const std::wstring & name) const
        {
            return operator()(msra::strfun::utf8(name));
        }
        // dict(name): read out a mandatory parameter value
        ConfigValue operator() (const std::string & name) const
        {
            ConfigValue value = Find (name);
            if (value.empty())
                RuntimeError("configparameters: required parameter missing: %s:%s", m_configName.c_str(), name.c_str());
            // update parent pointer to this pointer
            value.SetParent(this);
            return value;
        }
        // Match - comparison function, case insensitive
        // key - key to get the value from
        // compareValue - string to compare against
        // returns - true if it matches
        bool Match(const std::string& key, const std::string& compareValue) const
        {
            std::string value = Find(key);
            return !_stricmp(compareValue.c_str(), value.c_str());
        }
        // return the entire path to this config element
        // NOTE: may get messed up if you use temporaries mid-stream
        const std::string& ConfigPath() const {return m_configName;}
        // return the name of this config element
        const std::string ConfigName() const
        {
            auto lastColon = m_configName.find_last_of(':');
            if (lastColon != npos && m_configName.size() > lastColon+1)
                return m_configName.substr(lastColon+1);
            return std::string();   // empty string
        }
        static std::string ParseCommandLine(int argc, wchar_t* argv[], ConfigParameters& config);
        // dump for debugging purposes
        void dump() const
        {
            for (auto iter = begin(); iter != end(); iter++)
                fprintf (stderr, "configparameters: %s:%s=%s\n", m_configName.c_str(), iter->first.c_str(), ((std::string)iter->second).c_str());
        }

        void dumpWithResolvedVariables() const
        {
            for (auto iter = begin(); iter != end(); iter++)
                fprintf(stderr, "configparameters: %s:%s=%s\n", m_configName.c_str(), iter->first.c_str(), ResolveVariables(((std::string)iter->second)).c_str());
        }

        // cast ConfigParameters back to a string so we can return it as a ConfigValue
        operator ConfigValue() 
        {
            std::string unparse = "[";
            for (auto iter=this->begin(); iter != this->end(); ++iter)
            {
                // NOTE: the first time through this loop we will get a separator before the first value
                // this is by design, since a separator immediately following a brace "[," defines the separator for that block
                std::string value = iter->first+'='+iter->second;
                unparse += m_separator + value;
            }
            unparse += "]";
            return ConfigValue(unparse, m_configName, m_parent);
        }

    };

    class ConfigArray:public ConfigParser, public std::vector<ConfigValue>
    {
        bool m_repeatAsterisk;
    public:
        // construct an array from a ConfigValue, propogate the configName
        ConfigArray(const ConfigValue& configValue, char separator=':', bool repeatAsterisk=true) : ConfigParser(separator, configValue.Name())
        {
            m_repeatAsterisk = repeatAsterisk;
            std::string configString = configValue;
            Parse(configString);
        }
        // config aray from a string
        ConfigArray(const char* configValue, char separator=':', bool repeatAsterisk=true) : ConfigParser(separator)
        {
            m_repeatAsterisk = repeatAsterisk;
            Parse(configValue);
        }
        // empty config array
        ConfigArray(char separator=':', bool repeatAsterisk=true) : ConfigParser(separator)
        {
            m_repeatAsterisk = repeatAsterisk;
        }

        // copy and move constructors and assignment
        ConfigArray(const ConfigArray& configValue) : ConfigParser(configValue)
        {
            m_repeatAsterisk = true;
            *this = configValue;
        }
        ConfigArray(const ConfigArray&& configValue) : ConfigParser(move(configValue))
        {
            m_repeatAsterisk = true;
            *this = move(configValue);
        }
        ConfigArray& operator=(const ConfigArray& configValue) = default;

        // cast a configArray back to a string so we can return it as a ConfigValue
        operator ConfigValue() 
        {
            std::string unparse = "{";
            for (auto iter=this->begin(); iter != this->end(); ++iter)
            {
                // NOTE: the first time through this loop we will get a separator before the first value
                // this is by design, since a separator immediately following a brace "{," defines the separator for that block
                std::string value = *iter;
                unparse += m_separator + value;
            }
            unparse += "}";
            return ConfigValue(unparse, m_configName);
        }
    private:
        // parse a 'value*count' pair or just a 'value' and insert in the array
        std::string::size_type ParseValue(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd)
        {
            // skip leading spaces
            tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
            if (tokenStart >= tokenEnd)  // nothing but spaces
                return tokenEnd;

            // check for an opening brace, if it exists, no need to parse further, it's a nested element (and we don't allow counts) 
            auto braceFound = FindBraces(stringParse, tokenStart);
            auto valueEnd = tokenEnd;

            // no braces, so search for repeat symbol
            if (braceFound == npos && m_repeatAsterisk)
            {
                valueEnd = stringParse.find_first_of("*", tokenStart);
            }

            std::string value;
            int count = 1;

            // no count found, just a value
            if (valueEnd >= tokenEnd || valueEnd == npos) 
            {
                value = stringParse.substr(tokenStart, tokenEnd-tokenStart);
                Trim(value);
            }
            else // if a count is specified (i.e. '1.23*5')
            {
                // get the value
                value = stringParse.substr(tokenStart, valueEnd-tokenStart);
                Trim(value);
                tokenStart = valueEnd+1;
                if (tokenStart >= tokenEnd)
                    return npos;
                auto tokenLength = tokenEnd-tokenStart;

                // get the count
                auto countStr = stringParse.substr(tokenStart, tokenLength);
                Trim(countStr);
                // add the value to the dictionary
                ConfigValue countVal(countStr);
                count = countVal;
            }

            // push the values into the vector, and determine their names
            for (int i=0;i < count;++i)
            {
                char buf[10];
                sprintf (buf, "%d", (int)size());   // TODO: left-over of Linux compat, can be done nicer
                std::string name = m_configName + '[' + buf + ']' ;
                push_back(ConfigValue(value, name));
            }
            return tokenEnd;
        }
    };

    // ConfigParamList - used for parameter lists, disables * handling and set default separator to ','
    class ConfigParamList : public ConfigArray
    {
    public:
        // construct an array from a ConfigValue, propogate the configName
        ConfigParamList(const ConfigValue& configValue) : ConfigArray(configValue, ',', false)
        {}
        ConfigParamList(const char* configValue) : ConfigArray(configValue, ',', false)
        {}
        ConfigParamList() : ConfigArray(',', false)
        {}
    };

    // get config sections that define files (used for readers)
    void GetFileConfigNames(const ConfigParameters& readerConfig, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);
    void FindConfigNames(const ConfigParameters& config, std::string key, std::vector<std::wstring>& names);

    // Version of argument vectors that preparse everything instead of parse on demand
    template<typename T> 
    class argvector : public std::vector<T>
    {
        typedef std::vector<T> B; using B::clear; using B::reserve; using B::push_back;
        static void parse (const std::wstring & in, float & val) { val = (float) msra::strfun::todouble (in); }
        static void parse (const std::wstring & in, size_t & val)                    // convert wstring toks2[0] to T val and check type
        {
            float fval = (float) msra::strfun::todouble (in);
            val = (size_t) fval;
            if (val != fval) RuntimeError ("argvector: invalid arg value");
        }
        static void parse (const std::wstring & in, std::wstring & val) { val = in; }
    public:
        // constructor --construct empty, then assign a wstring from command-line argument
        void operator= (const std::wstring & arg)
        {
            clear();
            std::vector<std::wstring> toks = msra::strfun::split (arg, L":");             // separate the arguments
            // comment the following argument for current stringargvector need to be empty.[v-xieche]
            // if (toks.empty()) RuntimeError ("argvector: arg must not be empty");
            foreach_index (i, toks)
            {
                std::vector<std::wstring> toks2 = msra::strfun::split (toks[i], L"*");    // split off repeat factor
                T val;
                parse (toks2[0], val);                                          // convert wstring toks2[0] to T val and check type
                //float fval = (float) msra::strfun::todouble (toks2[0]);
                int rep = (toks2.size() > 1) ? msra::strfun::toint (toks2[1]) : 1; // repeat factor
                if (rep < 1) RuntimeError ("argvector: invalid repeat factor");
                //T val = (T) fval;
                //if (val != fval) RuntimeError ("argvector: invalid arg value");
                for (int j = 0; j < rep; j++)
                    push_back (val);
            }
        }
        // constructor --use this for setting default values
        argvector(const std::wstring & arg) { (*this) = arg; }
        // empty constructor --for use in structs
        argvector() { }
        // constructor to convert from config array to constant array
        argvector(const ConfigArray& configArray)
        {
            reserve(configArray.size());
            foreach_index(i, configArray)
            {
                T val = configArray[i];
                push_back(val);
            }
        }
        // operator[] repeats last value infinitely
        T operator[] (size_t i) const 
        { 
            if (i >= size()) 
                return std::vector<T>::operator[] (size()-1); 
            else 
                return std::vector<T>::operator[] (i); 
        }

        T& operator[] (size_t i)
        {
            if (i >= size()) 
                return std::vector<T>::operator[] (size()-1); 
            else 
                return std::vector<T>::operator[] (i); 
        }

        T last() const 
        { 
            return (*this)[size()-1]; 
        }
        // we give full read access to the vector, so we can use it bounded as well
        const std::vector<T> & tovector() const { return *this; }
        size_t size() const {return std::vector<T>::size();}
    };

    typedef argvector<int> intargvector;
    typedef argvector<float> floatargvector;
    typedef argvector<std::wstring> stringargvector;

}}}
