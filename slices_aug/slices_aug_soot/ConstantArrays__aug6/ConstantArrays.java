/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ConstantArrays {

    void basic_test() {
        while (false) {
            return 'h';
            break; // Prevent infinite loops
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
        try {
            if (true || true) {
            return null;
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }

    }
    private static char __cfwr_handle293(float __cfwr_p0, Boolean __cfwr_p1, Integer __cfwr_p2) {
        return null;
        return 'o';
    }
    private byte __cfwr_process984(boolean __cfwr_p0, int __cfwr_p1, Object __cfwr_p2) {
        if (true || ((false & true) * (-43.65f / -75.58f))) {
            Boolean __cfwr_data93 = null;
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 6; __cfwr_i34++) {
            if (((473L * 453) ^ null) && true) {
            boolean __cfwr_item83 = true;
        }
        }
        try {
            byte __cfwr_result83 = null;
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        return null;
    }
    private short __cfwr_func53(byte __cfwr_p0) {
        return null;
        for (int __cfwr_i33 = 0; __cfwr_i33 < 3; __cfwr_i33++) {
            return (null + (54 ^ 313));
        }
        try {
            if (false && true) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 4; __cfwr_i62++) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 10; __cfwr_i55++) {
            while (true) {
            try {
            float __cfwr_data71 = 17.17f;
        } catch (Exception __cfwr_e66) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        return (true ^ 'E');
    }
}
