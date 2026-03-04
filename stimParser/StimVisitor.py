# Generated from Stim.g4 by ANTLR 4.13.2
from antlr4 import *
import sys;


if "." in __name__:
    from .StimParser import StimParser
else:
    from StimParser import StimParser

import pyzx as zx
from fractions import Fraction

# This class defines a complete generic visitor for a parse tree produced by StimParser.

class StimVisitor(ParseTreeVisitor):
    circuit = None
    rec = []
    verbose = False
    lenient = False
    qubit_map = []
    tot_num = 0
    errors = []

    def _debugLog(self, msg):
        if self.verbose:
            print(msg)

    # Visit a parse tree produced by StimParser#circuit.
    def visitCircuit(self, ctx:StimParser.CircuitContext):
        self._debugLog("circuit visit")
        self.tot_num = 0
        self.circuit = zx.Circuit(self.tot_num)
        temp = [True] * self.tot_num
        # zx.Circuit.initialize_qubits(self.circuit, temp)
        self.qubit_map = [-1] * 600
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#line.
    def visitLine(self, ctx:StimParser.LineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#line_missing_newline.
    def visitLine_missing_newline(self, ctx:StimParser.Line_missing_newlineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#instruction.
    def visitInstruction(self, ctx:StimParser.InstructionContext):


        name = ctx.NAME().getText()  # e.g. "H", "CX", "M"
        self._debugLog(f"instruction visit {name}")
        tag = ctx.TAG().getText() if ctx.TAG() else None
        args = self.visit(ctx.parens_arguments()) if ctx.parens_arguments() else []
        targets = self.visit(ctx.targets())

        self.emit_gate(name, targets, args, tag)

        return None


    # Visit a parse tree produced by StimParser#parens_arguments.
    def visitParens_arguments(self, ctx:StimParser.Parens_argumentsContext):
        self._debugLog("parens visit")
        return self.visit(ctx.arguments())

    # Visit a parse tree produced by StimParser#arguments.
    def visitArguments(self, ctx:StimParser.ArgumentsContext):
        self._debugLog("arguments visit")
        vals = [float(ctx.arg().getText())]
        if ctx.arguments():
            vals.extend(self.visit(ctx.arguments()))
        return vals


    # Visit a parse tree produced by StimParser#targets.
    def visitTargets(self, ctx:StimParser.TargetsContext):
        self._debugLog("targets visit")
        all_targs = []
        for i in ctx.getChildren():
            all_targs.append(self.visit(i))

        return all_targs

    # Visit a parse tree produced by StimParser#arg.
    def visitArg(self, ctx:StimParser.ArgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#targ.
    def visitTarg(self, ctx:StimParser.TargContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#qubit_target.
    def visitQubit_target(self, ctx:StimParser.Qubit_targetContext):
        self._debugLog("qubit targ visit")
        inverted = ctx.getText().startswith('!')
        q = int(ctx.UINT().getText())
        return ["qubit", q, inverted]


    # Visit a parse tree produced by StimParser#measurement_record_target.
    def visitMeasurement_record_target(self, ctx:StimParser.Measurement_record_targetContext):
        self._debugLog("measurement targ visit")
        q = int(ctx.UINT().getText())
        return ["record", q]


    # Visit a parse tree produced by StimParser#sweep_bit_target.
    def visitSweep_bit_target(self, ctx:StimParser.Sweep_bit_targetContext):
        self._debugLog("sweep targ visit")
        q = int(ctx.UINT().getText())
        return ["sweep", q, False]


    # Visit a parse tree produced by StimParser#pauli_target.
    def visitPauli_target(self, ctx:StimParser.Pauli_targetContext):
        self._debugLog("pauli targ visit")
        targ = ctx.getText()
        inverted = targ.startswith('!')
        qubit = ctx.PAULI().getText()
        pauli = qubit[0]
        q = int(qubit[1:])
        return ["pauli", q, inverted, pauli]

    # Visit a parse tree produced by StimParser#combiner_target.
    def visitCombiner_target(self, ctx:StimParser.Combiner_targetContext):
        self._debugLog("combiner targ visit")
        return "*"


    # Visit a parse tree produced by StimParser#block_start.
    def visitBlock_start(self, ctx:StimParser.Block_startContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by StimParser#block_end.
    def visitBlock_end(self, ctx:StimParser.Block_endContext):
        return self.visitChildren(ctx)


    def emit_gate(self, name, targets, args, tag):
        if name == "I":
            return
        elif name == "X":
            for i in targets:
                self.circuit.add_gate("NOT", self.qubit_map[i[1]])
        elif name == "Y":
            for i in targets:
                self.circuit.add_gate("Y", self.qubit_map[i[1]])
        elif name == "Z":
            for i in targets:
                self.circuit.add_gate("Z", self.qubit_map[i[1]])

        elif name == "CX":
            for index, targ in enumerate(targets):
                if index % 2 == 0:
                    qubit1 = targets[index][1]
                    qubit2 = targets[index+1][1]
                    self._debugLog(f"cnot targets: {type(qubit1)}, {qubit2}")
                    self.circuit.add_gate("CNOT", self.qubit_map[qubit1],self.qubit_map[qubit2])

        elif name == "R":
            for i in targets:
                qubit = i[1]

                if self.qubit_map[qubit] == -1:
                    self.qubit_map[qubit] = self.tot_num
                    print(f"start R {self.qubit_map[qubit]}")
                    self.circuit.add_gate("InitAncilla", self.qubit_map[qubit])
                    self.circuit.add_gate("H", self.qubit_map[qubit])
                    self.tot_num += 1
                else:
                    print(f"mid R {self.qubit_map[qubit]}")
                    self.circuit.add_gate("Reset", self.qubit_map[qubit])

        elif name == "RX":
            for i in targets:
                qubit = i[1]
                if self.qubit_map[qubit] == -1:
                    self.qubit_map[qubit] = self.tot_num
                    print(f"start RX {self.qubit_map[qubit]}")
                    self.circuit.add_gate("InitAncilla", self.qubit_map[qubit])
                    self.tot_num += 1
                else:
                    print(f"mid RX {self.qubit_map[qubit]}")
                    self.circuit.add_gate("Reset", self.qubit_map[qubit])
                    self.circuit.add_gate("H", self.qubit_map[qubit])

        elif name == "M" or name == "MZ": #Projects each target qubit into |0> or |1> and reports its value (false=|0>, true=|1>
            for i in targets:
                qubit = i[1]
                inverted = i[2]
                val = len(self.rec)
                a = zx.symbolic.new_var(f"z{val}", is_bool=True)
                m = zx.gates.Measurement(self.qubit_map[qubit], result_symbol=a)

                self.circuit.add_gate(m)
                self.circuit.add_gate("InitAncilla", self.tot_num)
                self.circuit.add_gate("H", self.tot_num)
                self.circuit.add_gate("XPhase", self.tot_num, a)
                self.rec.append(a)
                self.qubit_map[qubit] = self.tot_num
                self.tot_num += 1


        elif name == "MX": #Projects each target qubit into |0> or |1> and reports its value (false=|0>, true=|1>
            for i in targets:
                qubit = i[1]
                inverted = i[2]
                val = len(self.rec)
                a = zx.symbolic.new_var(f"x{val}", is_bool=True)
                m = zx.gates.Measurement(self.qubit_map[qubit], result_symbol=a)

                self.circuit.add_gate(m)
                self.circuit.add_gate("InitAncilla", self.tot_num)
                self.circuit.add_gate("ZPhase", self.tot_num, a)
                self.rec.append(a)
                self.qubit_map[qubit] = self.tot_num
                self.tot_num += 1

        elif name == "S_DAG": #Replaces S dagger gates with T dagger
            for i in targets:
                self.circuit.add_gate("ZPhase", self.qubit_map[i[1]], phase=Fraction(7, 4))

        elif name == "S": #Replaces S dagger gates with T dagger
            for i in targets:
                self.circuit.add_gate("T", self.qubit_map[i[1]])

        elif name == "DEPOLARIZE1":
                for i in targets:
                    if self.qubit_map[i[1]] != -1:
                        a = zx.symbolic.new_var(f"{len(self.errors)}_{0}", is_bool=True)
                        b = zx.symbolic.new_var(f"{len(self.errors)}_{1}", is_bool=True)
                        self.circuit.add_gate("XPhase", self.qubit_map[i[1]], a)
                        self.circuit.add_gate("ZPhase", self.qubit_map[i[1]], b)
                        self.errors.append((a,b,"depolarize1"))

        elif name == "DEPOLARIZE2":
            for index, targ in enumerate(targets):
                if index % 2 == 0:
                    qubit1 = targets[index][1]
                    qubit2 = targets[index + 1][1]
                    a = zx.symbolic.new_var(f"{len(self.errors)}_{0}", is_bool=True)
                    b = zx.symbolic.new_var(f"{len(self.errors)}_{1}", is_bool=True)
                    self.circuit.add_gate("XPhase", self.qubit_map[qubit1], a)
                    self.circuit.add_gate("ZPhase", self.qubit_map[qubit1], b)
                    c = zx.symbolic.new_var(f"{len(self.errors)}_{2}", is_bool=True)
                    d = zx.symbolic.new_var(f"{len(self.errors)}_{3}", is_bool=True)
                    self.circuit.add_gate("XPhase", self.qubit_map[qubit2], c)
                    self.circuit.add_gate("ZPhase", self.qubit_map[qubit2], d)
                    self.errors.append((a,b,c,d,"depolarize2"))

        elif name == "Z_ERROR":
            for i in targets:
                a = zx.symbolic.new_var(f"z{len(self.errors)}", is_bool=True)
                self.circuit.add_gate("ZPhase", self.qubit_map[i[1]], a)
                self.errors.append((a,"z_error"))

        elif name == "X_ERROR":
            for i in targets:
                a = zx.symbolic.new_var(f"x{len(self.errors)}", is_bool=True)
                self.circuit.add_gate("XPhase", self.qubit_map[i[1]], a)
                self.errors.append((a,"x_error"))

        elif name == "TICK" or name == "DEPOLARIZE1" or name == "DEPOLARIZE2" or name == "X_ERROR" or name == "QUBIT_COORDS" or name == "DETECTOR" or name == "OBSERVABLE_INCLUDE" or name == "MPP" or name == "Z_ERROR":
            return


        else:
            if self.lenient:
                print(f"unimplemented gate: {name}")
            else:
                raise NotImplementedError(f"STIM gate {name}")

del StimParser