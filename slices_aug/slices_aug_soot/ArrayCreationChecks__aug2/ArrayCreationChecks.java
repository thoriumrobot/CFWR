/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
        if ((null | -97.36) && true) {
            if (false && true) {
            while (false) {
            return true;
            break; // Prevent infinite loops
        }
        }
        }

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
    int __cfwr_func883(Long __cfwr_p0, byte __cfwr_p1, float __cfwr_p2) {
        if (true && true) {
            return null;
        }
        if (false && true) {
            boolean __cfwr_entry7 = ('u' % -95.71);
        }
        while (true) {
            Character __cfwr_entry35 = null;
            break; // Prevent infinite loops
        }
        return 306;
    }
}
