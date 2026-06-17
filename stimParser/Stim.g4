grammar Stim;

circuit: line* line_missing_newline EOF;

line : line_missing_newline '\n';
line_missing_newline : (instruction | block_start | block_end)? COMMENT?;

instruction : NAME TAG? parens_arguments? targets;

parens_arguments : '(' arguments ')';
arguments : arg (',' arguments)?;

targets : targ*;

arg : UINT | DOUBLE;

targ : qubit_target | measurement_record_target | sweep_bit_target | pauli_target | combiner_target;

qubit_target : '!'? UINT;
measurement_record_target : 'rec[-' UINT ']';
sweep_bit_target : 'sweep[' UINT ']';
pauli_target : '!'? PAULI;
combiner_target : '*';

block_start : instruction '{';
block_end : '}';

INDENT: (' ' | '\r' | '\t')+ -> skip;
TAG : '[' ~[\r\n\]]* ']';
PAULI : ('X' | 'Y' | 'Z') UINT;

COMMENT
    : '#' .*? -> skip
    ;

NAME: IDENTIFIER;

UINT : INTEGER;
DOUBLE : INTEGER | FLOAT_NUMBER;

fragment INTEGER         : DECIMAL_INTEGER | OCT_INTEGER | HEX_INTEGER | BIN_INTEGER;
fragment DECIMAL_INTEGER : NON_ZERO_DIGIT DIGIT* | '0';
fragment OCT_INTEGER     : '0' ('o' | 'O') OCT_DIGIT+ | '0' OCT_DIGIT+;
fragment HEX_INTEGER     : '0' ('x' | 'X') HEX_DIGIT+;
fragment BIN_INTEGER     : '0' ('b' | 'B') BIN_DIGIT+;
fragment NON_ZERO_DIGIT  : [1-9];
fragment OCT_DIGIT       : [0-7];
fragment BIN_DIGIT       : '0' | '1';
fragment HEX_DIGIT       : DIGIT | [a-f] | [A-F];

fragment FLOAT_NUMBER   : POINT_FLOAT | EXPONENT_FLOAT;
fragment POINT_FLOAT    : INT_PART? FRACTION | INT_PART '.';
fragment EXPONENT_FLOAT : (INT_PART | POINT_FLOAT) EXPONENT;
fragment INT_PART       : DIGIT+;
fragment FRACTION       : '.' DIGIT+;
fragment EXPONENT       : ('e' | 'E') ('+' | '-')? DIGIT+;

fragment IDENTIFIER : LETTER (LETTER | DIGIT | '_')*;
fragment LETTER     : LOWERCASE | UPPERCASE;
fragment LOWERCASE  : [a-z];
fragment UPPERCASE  : [A-Z];
fragment DIGIT      : [0-9];



