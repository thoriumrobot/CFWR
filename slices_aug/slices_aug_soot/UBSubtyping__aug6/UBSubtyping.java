/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        long __cfwr_entry31 = 500L;

        @LTEqLengthOf({ "arr" })
        int a = 1;
        @LTLengthOf({ "arr" })
        int a1 = 1;
        @LTLengthOf({ "arr" })
        int b = a;
        @UpperBoundUnknown
        int d = a;
        @LTLengthOf({ "arr2" })
        int g = a;
        @LTEqLengthOf({ "arr", "arr2", "arr3" })
        int h = 2;
        @LTEqLengthOf({ "arr", "arr2" })
        int h2 = test;
        @LTEqLengthOf({ "arr" })
        int i = test;
        @LTEqLengthOf({ "arr", "arr3" })
        int j = test;
    }
  
        return 116;
  private Character __cfwr_temp36() {
        try {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 1; __cfwr_i78++) {
            if ((null ^ 'S') && (-230L + 726L)) {
            if (true || true) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 9; __cfwr_i28++) {
            while (('t' - true)) {
            Boolean __cfwr_val92 = null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        boolean __cfwr_elem19 = true;
        return null;
    }
}
