/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ConstantArrays {

    void basic_test() {
        try {
            return null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }

        int[] b = new int[4];
        @LTLengthOf("b")
        int[] a = { 0, 1, 2, 3 };
        @LTLengthOf("b")
        int[] a1 = { 0, 1, 2, 4 };
        @LTEqLengthOf("b")
        int[] c = { -1, 4, 3, 1 };
        @LTEqLengthOf("b")
        int[] c2 = { -1, 4, 5, 1 };
    }
    static String __cfwr_handle300(double __cfwr_p0) {
        for (int __cfwr_i56 = 0; __cfwr_i56 < 6; __cfwr_i56++) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 1; __cfwr_i45++) {
            while (false) {
            Boolean __cfwr_item61 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        return "value27";
    }
}
