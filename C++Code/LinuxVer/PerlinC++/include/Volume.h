#ifndef VOLUME_H__
#define VOLUME_H__

#include <vector>
#include "Utils.h"

namespace npy
{

template <typename _Tp>
class Volume
{

public:
    Volume( )
    {
        m_shape = Coordinate( 0,0,0 );
    }
    Volume( const Coordinate& shape )
    {
        m_data.resize( shape[0] *shape[1] *shape[2] );
        m_shape = shape;
        m_x_step = m_shape[1] *m_shape[2];
    }

    inline const size_t& getXStep() const { return m_x_step; }
    inline const _Tp& operator()( const size_t _x, const size_t _y, const size_t _z ) const { return operator()( Coordinate( _x,_y,_z ) ); }
    inline _Tp& operator()( const size_t _x, const size_t _y, const size_t _z ) { return operator()( Coordinate( _x,_y,_z ) ); }
    inline const _Tp& operator()( const Coordinate& _coord ) const
    {
        const size_t idx = m_x_step *_coord[0] +m_shape[2] *_coord[1] +_coord[2];
        return m_data[idx];
    }
    inline _Tp& operator()( const Coordinate& _coord )
    {
        const size_t idx = m_x_step *_coord[0] +m_shape[2] *_coord[1] +_coord[2];
        return m_data[idx];
    }

    inline void operator=( const Volume<_Tp>& _vol ) { m_data = _vol.data(); m_shape = _vol.shape(); m_x_step = _vol.getXStep(); }

    inline const _Tp& operator[]( const size_t _idx ) const { return m_data[_idx]; }
    inline _Tp& operator[]( const size_t _idx ) { return m_data[_idx]; }

    inline const Coordinate& shape() const { return m_shape; }
    inline const size_t& shape( const size_t _idx ) const { std::cout << "Getting Shape " << _idx << std::endl; return m_shape[_idx]; }

    inline const std::vector<_Tp>& data() const { return m_data; }
    inline std::vector<_Tp>& data() { return m_data; }

    inline void padFromVolume( const Volume<_Tp>& _input, const size_t _padding )
    {
        padFromVolume( _input, Coordinate( _padding, _padding, _padding ) );
    }

    inline void padFromVolume( const Volume<_Tp>& _input, const Coordinate& _padding )
    {
        m_shape = Coordinate( _input.shape()[0] + _padding[0]*2,
                              _input.shape()[1] + _padding[1]*2,
                              _input.shape()[2] + _padding[2]*2 );
        m_x_step = m_shape[1] *m_shape[2];
        std::cout << m_x_step << " <> " << _input.getXStep() << std::endl;

        m_data.clear();
        m_data.resize( m_shape[0] *m_shape[1] *m_shape[2] );

        Coordinate start_idx( _padding[0] ,_padding[1], _padding[2] );
        Coordinate end_idx( m_shape[0] -_padding[0] ,m_shape[1] -_padding[1], m_shape[2] -_padding[2] );

        size_t idx = 0;
        for( size_t x_it=start_idx[0]; x_it < end_idx[0]; ++x_it )
            for( size_t y_it=start_idx[1]; y_it < end_idx[1]; ++y_it )
            {
                //std::cout << "Copying " << x_it << ", " << y_it << " mem_idx: " << x_it *m_x_step +y_it *m_shape[2] +_padding[2] << std::endl;
                memcpy( reinterpret_cast<void*>( &( m_data.data()[x_it *m_x_step +y_it *m_shape[2] +_padding[2]] ) ), &_input( Coordinate( x_it-_padding[2],y_it-_padding[1],0 ) ), sizeof( _Tp ) *_input.shape()[2] );
            }
    }

    inline void addWithOffset( const Volume<_Tp>& _vol, const Coordinate& _offset )
    {
        for( size_t x=0; x < m_shape[0]; ++x )
            for( size_t y=0; y < m_shape[1]; ++y )
                for( size_t z=0; z < m_shape[2]; ++z )
                    operator()( x,y,z ) += _vol( x+_offset[0], y+_offset[1], z+_offset[2] );
    }

    inline std::string toString() const
    {
        std::stringstream sstr;
        for( size_t x=0; x < m_shape[0]; ++x )
        {
            sstr << "[";
            for( size_t y=0; y < m_shape[1]; ++y )
            {
                sstr << "[";
                for( size_t z=0; z < m_shape[2]; ++z )
                {
                    sstr << operator()( x,y,z );
                    if( z < m_shape[2] -1 ) sstr << ", ";
                }
                sstr << "]";
                if( y < m_shape[1] -1 ) sstr << ", ";
            }
            sstr << "]";
            if( x < m_shape[0] -1 ) sstr << ", ";
            sstr << std::endl;
        }
        return sstr.str();
    }

private:
    std::vector<_Tp> m_data;
    Coordinate m_shape;
    size_t m_x_step;
};

}

template <typename _Tp>
inline std::ostream& operator<<( std::ostream& _ostr, const npy::Volume<_Tp>& _obj )
{
    return _ostr << _obj.toString();
}

#endif // VOLUME_H__
