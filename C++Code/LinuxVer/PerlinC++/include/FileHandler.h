#ifndef NPY_FILE_HANDLER_H_
#define NPY_FILE_HANDLER_H_

#include <fstream>
#include <string>
#include <vector>
//#include <windows.h>
#include <exception>
#include <iostream>
#include <cerrno>
#include <cstring>

namespace npy
{

template <typename _Tp>
class FileHandler
{
public:
    FileHandler( const std::string& i_file_path, const std::string& o_file_path );
    FileHandler( const std::string& o_file_path );
    ~FileHandler();

    void loadFile( _Tp* data_buffer );
    void saveFile( _Tp* input, const size_t inp_size );

    inline const std::string& getOutPath() const
    {
        return m_o_path;
    }

private:
    std::string m_i_path, m_o_path;
};

}

#endif // NPY_FILE_HANDLER_H_
