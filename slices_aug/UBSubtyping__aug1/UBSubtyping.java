/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        short __cfwr_data97 = null;

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
        while (false) {
            if (true && false) {
            return -20.38f;
        }
            break; // Prevent infinite loops
        }

    static Integer __cfwr_aux862() {
        if (((null * -60.29) | 90.17f) && (null - (null % 31.39f))) {
            if (true || false) {
            return null;
        }
        }
        return null;
    }
}
