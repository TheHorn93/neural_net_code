struct ArgBase
{
  ArgBase( const std::string& new_arg, const bool init=false )
    : arg( new_arg ), is_init( init ) {}
  ArgBase( const std::string& new_arg, const int num, const bool init=false )
    : arg( new_arg ), num_params( num ), is_init( init ) {}

  inline bool operator==( const std::string& input ) const
  {
    return arg == input;
  }

  virtual void operator=( const std::string& new_val ) = 0;
  virtual std::string toString() const = 0;

  std::string arg;
  int num_params = 1;
  bool is_init = false;
};

template<class _Tp>
struct Arg : public ArgBase
{
  Arg( const std::string& new_arg ) : ArgBase( new_arg ) {}

  Arg( const std::string& new_arg, const int num_params )
    : ArgBase( new_arg, num_params ) {}

  Arg( const std::string& new_arg, const int num_params, const _Tp& def_val )
    : ArgBase( new_arg, num_params, true ), val( def_val ) {}

  void operator=( const std::string& new_val ) override;

  std::string toString() const override
  {
    std::stringstream sstr;
    sstr << arg << ": " << val;
    return sstr.str();
  }

  _Tp val;
};

template<>
void Arg<std::string>::operator=( const std::string& new_val ) { val = new_val; is_init = true; }

template<>
void Arg<int>::operator=( const std::string& new_val ) { val = stoi( new_val ); is_init = true; }

template<>
void Arg<double>::operator=( const std::string& new_val ) { val = stof( new_val ); is_init = true; }

template<>
void Arg<npy::Coordinate>::operator=( const std::string& new_val )
{
  std::string val_str = new_val;
  std::vector<std::string> params;
  size_t div;
  for( size_t it=0; it < 3; ++it )
  {
    div = val_str.find( "," );
    params.push_back( val_str.substr( 0, div ) );
    val_str = val_str.substr( div+1, val_str.npos );
  }
  if( params.size() != 3 )
    throw( std::invalid_argument( "Wrong number of arguments for a 3D-size" ) );
  val[0] = stoi( params[0] );
  val[1] = stoi( params[1] );
  val[2] = stoi( params[2] );
  is_init = true;
}

typedef std::unique_ptr<ArgBase> ArgPtr;

class InputArgs
{
public:
  InputArgs( int argc, char** input_strs )
  {
    addArg<std::string>( "-p" ); //output_path
    addArg<npy::Coordinate>( "-s", 3 ); //output_size
    addArg<int>( "-i" ); //iterations
    addArg<double>( "-f" ); //frequency
    addArg<double>( "-r", 1, 1.0 ); //range
    addArg<double>( "-d", 1, 2 ); //size/intensity divisor
    addArg<int>( "-t", 1, 4 ); //num threads
    for( int it=1; it < argc; it++ )
    {
      const std::string input = input_strs[it];
      int a_it = getArgIt( input );
      if( a_it > -1 )
        if( it < argc-1 )
        {
          try {
            if( args[a_it]->num_params == 1 )
              (*args[a_it]) = std::string( input_strs[it +1] );
            else
            {
              std::string new_arg = "";
              for( size_t s_it=1; s_it <= args[a_it]->num_params; ++s_it )
                new_arg += std::string(input_strs[it +s_it]) +std::string(",");
              (*args[a_it]) = new_arg;
            }
            it += args[a_it]->num_params;
          }
          catch( std::invalid_argument e ) {
            throw( std::invalid_argument( input + ": " +e.what() ) );
          }
        }
        else
          throw( std::invalid_argument( "Missing Argument for: " +input ) );
      else
        throw( std::invalid_argument( "Invalid parameter argument: " +input ) );
    }
    std::string missing_params = "";
    for( const ArgPtr& arg : args )
      if( !arg->is_init )
        missing_params += arg->arg +std::string( ", " );
    if( missing_params != "" )
      throw( std::invalid_argument( "Missing parameters: " +missing_params ) );
  }

  template<class _Tp>
  void addArg( const std::string& new_arg )
  {
    std::unique_ptr<Arg<_Tp>> arg( new Arg<_Tp>( new_arg ) );
    args.push_back( std::move( arg ) );
  }

  template<class _Tp>
  void addArg( const std::string& new_arg, const int num_params )
  {
    std::unique_ptr<Arg<_Tp>> arg( new Arg<_Tp>( new_arg, num_params ) );
    args.push_back( std::move( arg ) );
  }

  template<class _Tp>
  void addArg( const std::string& new_arg, const int num_params, const _Tp& def_val )
  {
    std::unique_ptr<Arg<_Tp>> arg( new Arg<_Tp>( new_arg, num_params, def_val ) );
    args.push_back( std::move( arg ) );
  }

  const int getArgIt( const std::string& inp ) const
  {
    for( size_t it=0; it < args.size(); ++it )
      if( (*args[it]) == inp )
        return it;
    return -1;
  }

  template<class _Tp>
  const _Tp getArg( const std::string& inp ) const
  {
    int it = getArgIt( inp );
    if( it > -1 && args[it]->is_init )
    {
      Arg<_Tp>* cast_arg = dynamic_cast<Arg<_Tp>*>( args[it].get() );
      if( cast_arg != nullptr )
        return cast_arg->val;
    }
    throw( std::invalid_argument( "Could not find initialized argument: " +inp ) );
  }

  std::string toString() const
  {
    std::stringstream sstr;
    for( size_t it=0; it < args.size(); ++it )
      sstr << (*args[it]).toString() << std::endl;
    return sstr.str();
  }

private:
  std::vector<ArgPtr> args;
};

template<class _Tp>
inline std::ostream& operator<<( std::ostream& _ostr, const Arg<_Tp>& _obj )
{
    return _ostr << _obj.toString();
}

template<class _Tp>
inline std::ostream& operator<<( std::ostream& _ostr, const ArgBase& _obj )
{
    return _ostr << _obj.toString();
}

