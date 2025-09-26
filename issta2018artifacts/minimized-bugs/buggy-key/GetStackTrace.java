// a simplified version of the bug fixed by plume-lib revision 7a6f75bf5a8b72a6ae823d4bd29a43742eb5cf50

public class GetStackTrace {
	/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public void error() {
	ThrowableT t = new ThrowableT();
        t.fillInStackTrace();
        StackTraceElementT[] ste = t.getStackTrace();
	//:: error: array.access.unsafe.high.constant
	StackTraceElementT caller = ste[1];
	printf(
			  "%s.%s (%s line %d)",
			  caller.getClassName(),
			  caller.getMethodName(),
			  caller.getFileName(),
			  caller.getLineNumber());
	for (int ii = 2; ii < ste.length; ii++) {
	    printf(" [%s line %d]", ste[ii].getFileName(), ste[ii].getLineNumber());
	}
    }

        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
        private static void printf(Object... s) {}

	private class StackTraceElementT {

		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public Object getClassName() {
			return null;
		}


		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public Object getMethodName() {
			return null;

		}


		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public Object getFileName() {
			return null;

		}


		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public Object getLineNumber() {			return null;

		}
	}

	private class ThrowableT {

		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public void fillInStackTrace() {

		}

		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public StackTraceElementT[] getStackTrace() {
			return null;
		}
	}
}
