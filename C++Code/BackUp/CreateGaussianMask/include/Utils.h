#ifndef NPY_UTILS_H_
#define NPY_UTILS_H_

#include <memory>
#include <time.h>
#include <cstdlib>
#include <exception>
#include <cmath>
#include <vector>
#include <sstream>
#include <ostream>

namespace npy
{

inline void strToNum( const std::string& _arg, size_t& _val )
{
    _val = static_cast<size_t>( std::stoi( _arg ) );
}

inline void strToNum( const std::string& _arg, double& _val )
{
    _val = std::stof( _arg );
}

template <typename _Tp>
struct Range
{
    Range() = default;

    Range( const _Tp par1, const _Tp par2 )
    {
        if ( par1 > par2 )
            throw std::runtime_error( "Range: Lower Border > Upper Border " );
        range[0] = par1;
        range[1] = par2;
    }

    inline void operator=( const std::string& _vals )
    {
        if( _vals[0] != '[' || _vals[_vals.size() -1] != ']' )
            throw std::runtime_error( "Range: fromString(): Wrong format. Expected \"[...]\" " );

        std::string lower_bound_str = _vals.substr( 1, _vals.find( ',' ) );
        std::string upper_bound_str = _vals.substr( _vals.find( ',' ) +1, _vals.npos -2 );
        strToNum( lower_bound_str, range[0] );
        strToNum( upper_bound_str, range[1] );
    }

    inline _Tp& operator[]( const size_t& _idx ) { return range[_idx]; }

    inline const _Tp& operator[]( const size_t& _idx ) const { return range[_idx]; }

    inline int getRandInt( ) const
    {
        int ran = range[1]-range[0]+1;
        return ( rand() % ran ) + range[0];
    }

    inline double getRandDouble( ) const
    {
        double ran = static_cast<double>(rand()) / RAND_MAX;
        return ran * (range[1]-range[0]) + range[0];
    }

    inline std::string toString() const
    {
        std::stringstream sstr;
        sstr << "[" << range[0] << "," << range[1] << "]";
        return sstr.str();
    }

private:
    _Tp range[2];
};


struct Coordinate
{
    Coordinate() = default;

    Coordinate( int x, int y, int z )
    {
        coor[0] = x;
        coor[1] = y;
        coor[2] = z;
    }

    Coordinate( const std::vector<int>& _input )
    {
        coor[0] = _input[0];
        coor[1] = _input[1];
        if( _input.size() == 3 )
            coor[2] = _input[2];
        else
            coor[2] = 0;
    }

    inline size_t getSqrdDistance( const Coordinate& _c ) const
    {
        return std::pow( coor[0]-_c.coor[0], 2 ) +
               std::pow( coor[1]-_c.coor[1], 2 ) +
               std::pow( coor[2]-_c.coor[2], 2 );
    }

    inline bool operator==( const Coordinate& _coord ) const
    {
        return  coor[0] == _coord.coor[0] &&
                coor[1] == _coord.coor[1] &&
                coor[2] == _coord.coor[2];
    }


    inline int& operator[]( const size_t _it )
    {
        return coor[_it];
    }

    inline const int& operator[]( const size_t _it ) const
    {
        return coor[_it];
    }

    inline void operator=( const std::string& _vals )
    {
        if( _vals[0] != '[' || _vals[_vals.size() -1] != ']' )
            throw std::runtime_error( "Range: fromString(): Wrong format. Expected \"[...]\" " );

        std::string proto_data_size = _vals.substr( 1, _vals.size() -1 );
        int it = 0; size_t pos = 0;
        do {
            pos = proto_data_size.find(",");
            std::string val = proto_data_size.substr(0, pos);
            coor[it] = std::stoi( val );
            ++it;
            proto_data_size.erase( 0, pos+1 );
        } while ( pos != std::string::npos );
    }

    inline std::string toString() const
    {
        std::stringstream sstr;
        sstr << "[" << coor[0] << "," << coor[1] << "," << coor[2] << "]";
        return sstr.str();
    }

    int coor[3];
};


struct Gaussian
{
    Coordinate center;
    double scale;
    double rad, sqrt_rad;
    double it_fac;

    Gaussian( Coordinate _center, double _scale, double _rad, double _max_rad )
    :
        center( _center ),
        scale( _scale ),
        rad( _rad )
    {;
        it_fac = _rad / _max_rad;
        sqrt_rad = std::pow( _rad, 2 );
    }
};

}

template<typename _Tp> inline std::ostream& operator<<( std::ostream& _ostr, const npy::Range<_Tp>& _obj )
{
    return _ostr << _obj.toString();
}

inline std::ostream& operator<<( std::ostream& _ostr, const npy::Coordinate& _obj )
{
    return _ostr << _obj.toString();
}

#endif // NPY_UTILS_H_
