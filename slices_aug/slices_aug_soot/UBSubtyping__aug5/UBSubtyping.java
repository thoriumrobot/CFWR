/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        if (false && false) {
            char __cfwr_data22 = 'l';
        }

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
    public int __cfwr_process335(char __cfwr_p0, Integer __cfwr_p1) {
        if (true || true) {
            if (true || true) {
            while ((null % (null % -51.81))) {
            return "result54";
            break; // Prevent infinite loops
        }
        }
        }
        while ((522L & true)) {
            while (false) {
            if (false || true) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 10; __cfwr_i33++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 4; __cfwr_i3++) {
            return null;
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        while (false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            if (true && true) {
            Double __cfwr_entry95 = null;
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        return 647;
    }
    public static Character __cfwr_compute617(int __cfwr_p0) {
        return (('O' << 15.99) + null);
        return null;
    }
}
