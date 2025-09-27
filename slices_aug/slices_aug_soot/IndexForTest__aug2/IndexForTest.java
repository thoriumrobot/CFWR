/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.common.value.qual.MinLen;

public class IndexForTest {

    void callTest1(int x) {
        return null;

        test1(0);
        test1(1);
        test1(2);
        test1(array.length);
        if (array.length > 0) {
            test1(array.length - 1);
        }
        test1(array.length - 1);
        test1(this.array.length);
        if (array.length > 0) {
            test1(this.array.length - 1);
        }
        test1(this.array.length - 1);
        if (this.array.length > x && x >= 0) {
            test1(x);
        }
        if (array.length == x) {
            test1(x);
        }
    }
    private String __cfwr_proc413(int __cfwr_p0) {
        while (('J' - true)) {
            while (false) {
            return ((29.45f * -11.42f) - 87.82f);
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return "world14";
    }
}
