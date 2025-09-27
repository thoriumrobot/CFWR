/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
        double __cfwr_temp91 = (32.47 % null);

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
    protected byte __cfwr_aux382() {
        if (true || true) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 7; __cfwr_i58++) {
            if (false && true) {
            while (false) {
            if (true || false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        try {
            while (true) {
            if ((507 ^ null) && false) {
            boolean __cfwr_item9 = false;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return null;
    }
    public Boolean __cfwr_calc755(Integer __cfwr_p0) {
        if ((null >> false) || (-77.79 | (-46L + null))) {
            try {
            Long __cfwr_elem37 = null;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        if (true && false) {
            return null;
        }
        Object __cfwr_temp54 = null;
        boolean __cfwr_elem12 = (-383 * (null & 'h'));
        return null;
    }
}
