#include "DataHandler.h"
namespace npy
{
#ifndef MAX_THREADS__
#define MAX_THREADS__ 4
#endif // MAX_THREADS__

#define FWHM 2.35482

inline void iterateCoordinate( const Coordinate& _size, Coordinate& _iter )
{
    if( _iter[2] < _size[2]-1 )
    {
        ++_iter[2];
    } else
    {
        _iter[2] = 0;
        if( _iter[1] < _size[1]-1 )
        {
            ++_iter[1];
        }
        else
        {
            _iter[1] = 0;
            ++_iter[0];
        }
    }
}

void evalGauss( const size_t _max_div, const double _iter, std::vector<double>& _output )
{
    size_t max_iter = _max_div * FWHM / _iter;
    std::vector<double> output;
    output.reserve( max_iter );

    float scale_iter = 0.0;
    float variance = std::pow( _max_div, 2 );
    for( size_t it=0; it < max_iter; ++it )
    {
        double expo = - std::pow( it *_iter, 2 ) / ( 2 * variance );
        output.push_back( exp( expo ) );
    }
    _output = output;
}

template <typename _Tp>
DataHandler<_Tp>::DataHandler( const std::string& _i_file_path, const std::string& _o_file_path, npy::Coordinate _shape  )
:
    m_shape( _shape ),
    m_size( 1 )
{
    for( size_t it=0; it < 3; ++it )
        m_size *= m_shape[it];
    m_data = new _Tp[ m_size ];
    m_f_handler.reset( new FileHandler<_Tp>( _i_file_path, _o_file_path ) );
    m_f_handler->loadFile( m_data );
}

template <typename _Tp>
DataHandler<_Tp>::DataHandler( const std::string& _o_file_path, npy::Coordinate _shape  )
:
    m_shape( _shape ),
    m_size( 1 )
{
    for( size_t it=0; it < 3; ++it )
        m_size *= m_shape[it];
    m_data = new _Tp[ m_size ];
    m_f_handler.reset( new FileHandler<_Tp>( _o_file_path ) );
    m_data = new _Tp[m_size];
    std::cout << "Allocated " << m_size << " elements" << std::endl;
}

template <typename _Tp>
DataHandler<_Tp>::~DataHandler()
{
    std::cout << "Freeing Buffer" << std::endl;
    delete[] m_data;
}

template <typename _Tp>
void DataHandler<_Tp>::addMultipleGaussians( const Range<size_t> _num_gauss,
                                             const Range<double> _scale_r,
                                             const Range<double> _std_div_r )
{
    srand( time( nullptr ) );
    size_t max_range = std::ceil( _std_div_r[1]*FWHM );
    double max_rad = _std_div_r[1] * FWHM;
    double g_idx = 0.1;
    std::vector<double> gauss;
    evalGauss( _std_div_r[1], g_idx, gauss );

    size_t num_gauss = _num_gauss.getRandInt();
    for( size_t it=0; it < num_gauss; ++it )
    {
        Coordinate center;
        do
        {
            std::vector<int> coord;
            for( int it=0; it < 3; ++it )
                coord.push_back( rand()%m_shape[it] );
            center = Coordinate( coord );
        }
        while( CoordInList( center ) );
        m_gaussians.push_back( Gaussian( center, _scale_r.getRandDouble(), _std_div_r.getRandDouble() * FWHM, max_rad ) );
        m_coords.push_back( center );
    }

    Coordinate voxel( 0, 0, 0 );
    for( size_t it=0; it < m_size; ++it )
    {
        for( size_t g_it=0; g_it < m_gaussians.size(); ++g_it )
        {
            size_t sq_diff = voxel.getSqrdDistance( ( m_gaussians[g_it] ).center );
            if ( sq_diff <= m_gaussians[g_it].sqrt_rad )
            {
                double proto_it = ( std::sqrt( sq_diff ) / m_gaussians[g_it].rad ) * max_range / g_idx;
                double weight = std::ceil( proto_it ) - proto_it;
                m_data[it] += m_gaussians[g_it].scale *( weight * gauss[std::floor( proto_it )] + (1-weight) * gauss[std::ceil( proto_it )] );
            }
        }
        if( m_data[it] > 1.0 )
            m_data[it] = 1.0;
        else if( m_data[it] < -1.0 )
            m_data[it] = -1.0;
        if( voxel.coor[1] == 410 && voxel.coor[2] == 0 )
            std::cout << voxel.toString() << ": " << it << "/" << m_size << std::endl;

        iterateCoordinate( m_shape, voxel );
    }

    m_f_handler->saveFile( m_data, m_size );
}

}

template class npy::DataHandler<float>;
template class npy::DataHandler<double>;
