/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroSub {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        boolean __cfwr_obj31 = false;

        @LTLengthOf({ "arr" })
        int c = a - (-1);
        @LTEqLengthOf({ "arr" })
        int c1 = a - (-1);
        @LTLengthOf({ "arr" })
        int d = a - 0;
        @LTLengthOf({ "arr" })
        int e = a - 7;
        @LTLengthOf({ "arr" })
        int f = a - (-7);
        @LTEqLengthOf({ "arr" })
        int j = 7;
    }
    public boolean __cfwr_process419(Integer __cfwr_p0) {
        for (int __cfwr_i75 = 0; __cfwr_i75 < 3; __cfwr_i75++) {
            return null;
        }
        try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return false;
    }
}
