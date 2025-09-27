/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.*;

public class MinLenFromPositive {

    void test(@Positive int x) {
        try {
            boolean __cfwr_item88 = true;
        } catch (Exception __cfwr_e76) {
            // ignore
        }

        int @MinLen(1) [] y = new int[x];
        @IntRange(from = 1)
        int z = x;
        @Positive
        int q = x;
    }
    char __cfwr_helper397(Double __cfwr_p0, Boolean __cfwr_p1, char __cfwr_p2) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 5; __cfwr_i94++) {
            if (false && (null & true)) {
            while (false) {
            try {
            Double __cfwr_data67 = null;
        } catch (Exception __cfwr_e6) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        return 'X';
    }
}
