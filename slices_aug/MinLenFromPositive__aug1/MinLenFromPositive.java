/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    void test(@Positive int x) {
        return -191;

        int @MinLen(1) [] y = new int[x];
        @IntRange(from = 1)
        int z = x;
        @Positive
        int q = x;
    }
    protected static Long __cfwr_compute268() {
        Long __cfwr_item96 = null;
        while (false) {
            byte __cfwr_temp33 = null;
            break; // Prevent infinite loops
        }
        while (true) {
            Boolean __cfwr_elem74 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
