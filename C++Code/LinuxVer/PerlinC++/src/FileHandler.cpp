#include "FileHandler.h"

namespace npy
{

template< typename _Tp >
FileHandler<_Tp>::FileHandler( const std::string& _i_file_path, const std::string& _o_file_path )
:
    m_i_path( _i_file_path ),
    m_o_path( _o_file_path )
{
    std::cout << "Input path: " << m_i_path << std::endl;
    std::cout << "Output path: " << m_o_path << std::endl;
}

template< typename _Tp >
FileHandler<_Tp>::FileHandler( const std::string& _o_file_path )
:
    m_o_path( _o_file_path )
{
    std::cout << "Output path: " << m_o_path << std::endl;
}

template< typename _Tp >
FileHandler<_Tp>::~FileHandler()
{

}

template< typename _Tp >
void FileHandler<_Tp>::loadFile( _Tp* data_buffer )
{
    size_t file_length = std::ifstream( m_i_path.c_str(), std::ios::ate | std::ios::binary ).tellg();
    int length = sizeof( _Tp );
    std::ifstream data( m_i_path.c_str(), std::ios::binary );
    data.seekg( 0, std::ios::beg );
    if( !data.is_open() )
        throw( std::runtime_error( "Error in reading file: \"" + m_i_path + "\"" ) );

    std::vector<_Tp> output;
    output.resize( file_length/length );
    //_Tp buffer;
    //for ( size_t it=0; it < file_length/length; ++it )
    //    data.read( reinterpret_cast<char*>( &output[it] ), length );
    data.read( reinterpret_cast<char*>( data_buffer ), file_length );

    data.close();
}

template< typename _Tp >
void FileHandler<_Tp>::saveFile( _Tp* _input, const size_t _inp_size )
{
    std::cout << "Writing to file" << std::endl;
    std::ofstream data_file( m_o_path, std::ios::out | std::ios::binary );
    data_file.write( reinterpret_cast<char*>( _input ), _inp_size*sizeof( _Tp ) );
    std::cout << "Writing Error: " << std::strerror( errno ) << std::endl;
    data_file.close();
}

}

template class npy::FileHandler<float>;
template class npy::FileHandler<double>;
