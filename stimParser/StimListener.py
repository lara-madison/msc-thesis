# Generated from Stim.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .StimParser import StimParser
else:
    from StimParser import StimParser

# This class defines a complete listener for a parse tree produced by StimParser.
class StimListener(ParseTreeListener):

    # Enter a parse tree produced by StimParser#circuit.
    def enterCircuit(self, ctx:StimParser.CircuitContext):
        pass

    # Exit a parse tree produced by StimParser#circuit.
    def exitCircuit(self, ctx:StimParser.CircuitContext):
        pass


    # Enter a parse tree produced by StimParser#line.
    def enterLine(self, ctx:StimParser.LineContext):
        pass

    # Exit a parse tree produced by StimParser#line.
    def exitLine(self, ctx:StimParser.LineContext):
        pass


    # Enter a parse tree produced by StimParser#line_missing_newline.
    def enterLine_missing_newline(self, ctx:StimParser.Line_missing_newlineContext):
        pass

    # Exit a parse tree produced by StimParser#line_missing_newline.
    def exitLine_missing_newline(self, ctx:StimParser.Line_missing_newlineContext):
        pass


    # Enter a parse tree produced by StimParser#instruction.
    def enterInstruction(self, ctx:StimParser.InstructionContext):
        pass

    # Exit a parse tree produced by StimParser#instruction.
    def exitInstruction(self, ctx:StimParser.InstructionContext):
        pass


    # Enter a parse tree produced by StimParser#parens_arguments.
    def enterParens_arguments(self, ctx:StimParser.Parens_argumentsContext):
        pass

    # Exit a parse tree produced by StimParser#parens_arguments.
    def exitParens_arguments(self, ctx:StimParser.Parens_argumentsContext):
        pass


    # Enter a parse tree produced by StimParser#arguments.
    def enterArguments(self, ctx:StimParser.ArgumentsContext):
        pass

    # Exit a parse tree produced by StimParser#arguments.
    def exitArguments(self, ctx:StimParser.ArgumentsContext):
        pass


    # Enter a parse tree produced by StimParser#targets.
    def enterTargets(self, ctx:StimParser.TargetsContext):
        pass

    # Exit a parse tree produced by StimParser#targets.
    def exitTargets(self, ctx:StimParser.TargetsContext):
        pass


    # Enter a parse tree produced by StimParser#arg.
    def enterArg(self, ctx:StimParser.ArgContext):
        pass

    # Exit a parse tree produced by StimParser#arg.
    def exitArg(self, ctx:StimParser.ArgContext):
        pass


    # Enter a parse tree produced by StimParser#targ.
    def enterTarg(self, ctx:StimParser.TargContext):
        pass

    # Exit a parse tree produced by StimParser#targ.
    def exitTarg(self, ctx:StimParser.TargContext):
        pass


    # Enter a parse tree produced by StimParser#qubit_target.
    def enterQubit_target(self, ctx:StimParser.Qubit_targetContext):
        pass

    # Exit a parse tree produced by StimParser#qubit_target.
    def exitQubit_target(self, ctx:StimParser.Qubit_targetContext):
        pass


    # Enter a parse tree produced by StimParser#measurement_record_target.
    def enterMeasurement_record_target(self, ctx:StimParser.Measurement_record_targetContext):
        pass

    # Exit a parse tree produced by StimParser#measurement_record_target.
    def exitMeasurement_record_target(self, ctx:StimParser.Measurement_record_targetContext):
        pass


    # Enter a parse tree produced by StimParser#sweep_bit_target.
    def enterSweep_bit_target(self, ctx:StimParser.Sweep_bit_targetContext):
        pass

    # Exit a parse tree produced by StimParser#sweep_bit_target.
    def exitSweep_bit_target(self, ctx:StimParser.Sweep_bit_targetContext):
        pass


    # Enter a parse tree produced by StimParser#pauli_target.
    def enterPauli_target(self, ctx:StimParser.Pauli_targetContext):
        pass

    # Exit a parse tree produced by StimParser#pauli_target.
    def exitPauli_target(self, ctx:StimParser.Pauli_targetContext):
        pass


    # Enter a parse tree produced by StimParser#combiner_target.
    def enterCombiner_target(self, ctx:StimParser.Combiner_targetContext):
        pass

    # Exit a parse tree produced by StimParser#combiner_target.
    def exitCombiner_target(self, ctx:StimParser.Combiner_targetContext):
        pass


    # Enter a parse tree produced by StimParser#block_start.
    def enterBlock_start(self, ctx:StimParser.Block_startContext):
        pass

    # Exit a parse tree produced by StimParser#block_start.
    def exitBlock_start(self, ctx:StimParser.Block_startContext):
        pass


    # Enter a parse tree produced by StimParser#block_end.
    def enterBlock_end(self, ctx:StimParser.Block_endContext):
        pass

    # Exit a parse tree produced by StimParser#block_end.
    def exitBlock_end(self, ctx:StimParser.Block_endContext):
        pass



del StimParser