/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        Long __cfwr_entry83 = null;

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
    static double __cfwr_compute51(String __cfwr_p0, Long __cfwr_p1, Boolean __cfwr_p2) {
        try {
            try {
            while (true) {
            if (false && true) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 5; __cfwr_i82++) {
            try {
            return 8.48;
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        try {
            float __cfwr_data97 = -45.13f;
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        return null;
        return null;
        return -85.75;
    }
    protected static String __cfwr_handle409() {
        while ((-322L * (null ^ -579))) {
            try {
            String __cfwr_temp30 = "test57";
        } catch (Exception __cfwr_e59) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return "temp25";
    }
}
