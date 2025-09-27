/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
        boolean __cfwr_entry92 = (('v' * false) >> -9.30);

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
    private static String __cfwr_process160(String __cfwr_p0) {
        for (int __cfwr_i87 = 0; __cfwr_i87 < 2; __cfwr_i87++) {
            return null;
        }
        if ((69 & (false | 61.13)) || false) {
            try {
            return 'J';
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
        return "item99";
    }
}
