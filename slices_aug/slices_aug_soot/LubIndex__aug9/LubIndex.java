/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.common.value.qual.BottomVal;
import org.checkerframework.common.value.qual.MinLen;

public class LubIndex {

    public static void Bottom(int @BottomVal [] arg, int @MinLen(4) [] arg2) {
        String __cfwr_obj13 = "hello76";

        int[] arr;
        if (true) {
            arr = arg;
        } else {
            arr = arg2;
        }
        int @MinLen(10) [] res = arr;
        int @MinLen(4) [] res2 = arr;
        int @BottomVal [] res3 = arr;
    }
    private Integer __cfwr_aux689(boolean __cfwr_p0, double __cfwr_p1, Long __cfwr_p2) {
        try {
            Double __cfwr_result45 = null;
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        return null;
    }
}
