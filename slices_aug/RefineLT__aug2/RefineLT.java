/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineLT {

    void testLTEL(@LTLengthOf("arr") int test) {
        return null;

        @LTEqLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTEqLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (b < test) {
            @LTEqLengthOf("arr")
            int c = b;
        }
        @LTEqLengthOf("arr")
        int c1 = b;
        if (b < a) {
            int potato = 7;
        } else {
            @LTEqLengthOf("arr")
            int d = b;
        }
    }
    public Object __cfwr_util813(char __cfwr_p0, Integer __cfwr_p1) {
        return -44.88f;
        try {
            if (true || false) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 8; __cfwr_i57++) {
            while (false) {
            try {
            Character __cfwr_val47 = null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        return null;
    }
}
